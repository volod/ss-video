"""Fast and deep profile orchestration."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from selfsuvis.pipeline.analysis4d.budget import BoundedFrameQueue, BudgetController
from selfsuvis.pipeline.analysis4d.geometry_providers import FailingDepth, ScriptedDepth
from selfsuvis.pipeline.analysis4d.io import read_jsonl, read_model
from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal
from selfsuvis.pipeline.analysis4d.profile import scheduled_analysis_jobs
from selfsuvis.pipeline.analysis4d.providers import Prompt, ScriptedGrounding
from selfsuvis.pipeline.analysis4d.publish import publish_accepted
from selfsuvis.pipeline.analysis4d.schemas import (
    AnalysisManifest,
    GapRecord,
    GeometrySample,
    SceneTimeline,
    TrackRecord,
    Verification,
)
from selfsuvis.pipeline.analysis4d.tracks import Detection
from selfsuvis.pipeline.analysis4d.verified_scene_event import VerifiedSceneEvent
from selfsuvis.pipeline.workflows.analysis4d_profile import (
    load_region_boxes,
    run_deep_profile,
    run_fast_profile,
)

_PROMPT = Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")
_WHEN = datetime(2026, 9, 22, 8, 0, tzinfo=UTC)


def _detection() -> Detection:
    return Detection(
        prompt_id="prompt-crate",
        label_raw="crate",
        label_normalized="crate",
        xywh=(0.2, 0.2, 0.2, 0.2),
        confidence=0.9,
        appearance=(1.0, 0.0),
    )


def _frames() -> list[FrameSignal]:
    frames = []
    for index in range(12):
        stamp = float(index)
        frames.append(
            FrameSignal(
                t_sec=stamp,
                embedding=(1.0, 0.0),
                drift=0.0,
                scene_cut=index in {0, 6},
                quality_ok=True,
            )
        )
    return frames


def _grounding() -> ScriptedGrounding:
    hit = _detection()
    return ScriptedGrounding({0.0: [hit], 6.0: [hit]})


def _grounding_at(stamp: float) -> ScriptedGrounding:
    return ScriptedGrounding({stamp: [_detection()]})


def _image_frames(directory: Path, count: int) -> list[FrameSignal]:
    path = directory / "frame.jpg"
    Image.new("RGB", (64, 48), (40, 90, 30)).save(path)
    frames = []
    for index in range(count):
        frames.append(
            FrameSignal(
                t_sec=float(index),
                embedding=(1.0, 0.0),
                scene_cut=index in {0, 6},
                quality_ok=True,
                image=str(path),
            )
        )
    return frames


def _write_regions(directory: Path) -> None:
    payload = {
        "schema_version": "ss-video.region-boxes.v1",
        "regions": [
            {
                "node_id": "region-yard",
                "label": "yard",
                "center_m": [0.0, 0.0, 1.0],
                "extent_m": [20.0, 20.0, 20.0],
            }
        ],
    }
    (directory / "regions.json").write_text(json.dumps(payload), encoding="utf-8")


def test_queue_keeps_order_and_covers_every_discard() -> None:
    queue = BoundedFrameQueue(2)
    dropped = []
    for frame in _frames():
        for span in queue.push(frame):
            assert span.start_sec <= span.dropped_t_sec < span.end_sec
            dropped.append(span.dropped_t_sec)
    assert queue.max_depth <= 2
    assert dropped
    kept = [frame.t_sec for frame in queue.drain()]
    assert kept == sorted(kept)


def test_tracks_are_admitted_when_optional_stages_are_shed() -> None:
    budget = BudgetController(gpu_slots=0)
    assert budget.admit("tracks")
    assert not budget.admit("vlm")
    assert budget.shed == ["vlm"]


def test_off_profile_schedules_no_jobs(monkeypatch) -> None:
    monkeypatch.delenv("ANALYSIS4D_PROFILE", raising=False)
    assert scheduled_analysis_jobs({}) == []
    assert scheduled_analysis_jobs({"analysis_profile": "off"}) == []
    assert scheduled_analysis_jobs({"analysis_profile": "fast"}) == ["analysis4d_fast"]
    assert scheduled_analysis_jobs({"analysis_profile": "deep"}) == [
        "analysis4d_fast",
        "postflight_analysis4d_deep",
    ]


def test_fast_publish_is_idempotent_and_deep_supersedes(tmp_path: Path) -> None:
    kwargs = {
        "grounding": _grounding(),
        "fixture_geometry": True,
        "capacity": 3,
        "slots": 0,
        "published_at": _WHEN,
        "duration_sec": 12.0,
    }
    first = run_fast_profile("mission-profile", _frames(), [_PROMPT], dest=tmp_path, **kwargs)
    assert first.queue_depth <= 3
    assert first.published_event_ids
    assert first.real_time_factor <= 1.0
    assert first.lags_sec
    assert max(first.lags_sec) <= 3.0
    gaps = [
        gap
        for gap in read_jsonl(tmp_path / "gaps.jsonl", GapRecord)
        if gap.reason == "queue_coalesce"
    ]
    assert gaps
    ledger = (tmp_path / "published-events.jsonl").read_text(encoding="utf-8")
    assert "rejected" not in ledger
    for line in ledger.splitlines():
        payload = VerifiedSceneEvent.model_validate(json.loads(line)["payload"])
        assert payload.verification_status == "accepted"

    second = run_fast_profile("mission-profile", _frames(), [_PROMPT], dest=tmp_path, **kwargs)
    assert second.replayed
    assert (tmp_path / "published-events.jsonl").read_text(encoding="utf-8") == ledger

    deep = run_deep_profile("mission-profile", _frames(), [_PROMPT], dest=tmp_path, **kwargs)
    assert deep.superseded
    timeline = read_model(tmp_path / "timeline.json", SceneTimeline)
    manifest = read_model(tmp_path / "manifest.json", AnalysisManifest)
    assert timeline.profile == "deep"
    assert manifest.profile == "deep"
    assert manifest.supersedes_sha256
    assert any(event.event_id in first.published_event_ids for event in timeline.events)
    assert (tmp_path / "history").is_dir()
    deep_ledger = (tmp_path / "published-events.jsonl").read_text(encoding="utf-8")
    again = run_deep_profile("mission-profile", _frames(), [_PROMPT], dest=tmp_path, **kwargs)
    assert again.replayed
    assert (tmp_path / "published-events.jsonl").read_text(encoding="utf-8") == deep_ledger


def test_admitted_geometry_cites_unavailable_boxes(tmp_path: Path) -> None:
    dest = tmp_path / "4d"
    _write_regions(tmp_path)
    result = run_fast_profile(
        "mission-geo",
        _image_frames(tmp_path, count=3),
        [_PROMPT],
        dest=dest,
        grounding=_grounding_at(0.0),
        depth=ScriptedDepth(np.full((48, 64), 4.0), "metric"),
        slots=1,
        capacity=32,
        fixture_geometry=False,
        published_at=_WHEN,
        duration_sec=3.0,
    )
    manifest = read_model(dest / "manifest.json", AnalysisManifest)
    assert manifest.coordinate_frame.metric_scale == "unavailable"
    assert manifest.coordinate_frame.calibration_id is None
    samples = [read_model(path, GeometrySample) for path in (dest / "geometry").rglob("*.json")]
    track_samples = [sample for sample in samples if sample.subject_id.startswith("fast-")]
    assert track_samples
    assert {sample.metric_scale for sample in track_samples} == {"unavailable"}
    assert all(sample.calibration_id is None for sample in track_samples)
    timeline = read_model(dest / "timeline.json", SceneTimeline)
    cited = [
        event
        for event in timeline.events
        if event.verification.status == "accepted"
        and event.evidence
        and event.evidence[0].geometry_ref
    ]
    assert cited
    assert any(event.type == "count_changed" for event in cited)
    assert result.published_event_ids
    ref = dest / cited[0].evidence[0].geometry_ref
    stored = read_model(ref, GeometrySample)
    assert stored.metric_scale == "unavailable"
    assert any((dest / "geometry" / "region-yard").glob("*.json"))


def test_spaced_label_publishes_a_count_event(tmp_path: Path) -> None:
    hit = Detection(
        prompt_id="prompt-card",
        label_raw="graphics card",
        label_normalized="graphics card",
        xywh=(0.2, 0.2, 0.2, 0.2),
        confidence=0.9,
        appearance=(1.0, 0.0),
    )
    prompt = Prompt(prompt_id="prompt-card", text="graphics card", normalized="graphics card")
    dest = tmp_path / "4d"
    result = run_fast_profile(
        "mission-card",
        _image_frames(tmp_path, count=2),
        [prompt],
        dest=dest,
        grounding=ScriptedGrounding({0.0: [hit]}),
        depth=ScriptedDepth(np.full((48, 64), 3.0), "relative"),
        slots=1,
        capacity=32,
        fixture_geometry=False,
        published_at=_WHEN,
        duration_sec=2.0,
    )
    assert result.published_event_ids
    assert all(" " not in event_id for event_id in result.published_event_ids)
    timeline = read_model(dest / "timeline.json", SceneTimeline)
    summaries = [event.summary for event in timeline.events if event.type == "count_changed"]
    assert any("graphics card" in summary for summary in summaries)


def test_saturated_queue_keeps_tracks_without_depth(tmp_path: Path) -> None:
    called: list[int] = []

    class _SpyDepth:
        failed = False
        scale = "relative"
        provider_id = "spy"

        def estimate(self, view: object) -> None:
            del view
            called.append(1)
            return None

    dest = tmp_path / "4d"
    result = run_fast_profile(
        "mission-queue",
        _image_frames(tmp_path, count=12),
        [_PROMPT],
        dest=dest,
        grounding=_grounding(),
        depth=_SpyDepth(),
        slots=1,
        capacity=3,
        fixture_geometry=True,
        published_at=_WHEN,
        duration_sec=12.0,
    )
    assert called == []
    assert "dense_geometry" in result.backlog
    assert "budget_shed" in result.degradations
    tracks = read_jsonl(dest / "tracks.jsonl", TrackRecord)
    assert tracks
    gaps = [
        gap for gap in read_jsonl(dest / "gaps.jsonl", GapRecord) if gap.reason == "queue_coalesce"
    ]
    assert gaps


def test_failed_depth_keeps_tracks(tmp_path: Path) -> None:
    dest = tmp_path / "4d"
    run_fast_profile(
        "mission-fail",
        _image_frames(tmp_path, count=2),
        [_PROMPT],
        dest=dest,
        grounding=_grounding_at(0.0),
        depth=FailingDepth(),
        slots=1,
        capacity=32,
        fixture_geometry=False,
        published_at=_WHEN,
        duration_sec=2.0,
    )
    manifest = read_model(dest / "manifest.json", AnalysisManifest)
    assert "provider_unavailable" in {item.code for item in manifest.degradations}
    assert read_jsonl(dest / "tracks.jsonl", TrackRecord)
    assert manifest.coordinate_frame.metric_scale == "unavailable"


def test_region_file_ignores_a_bad_schema(tmp_path: Path) -> None:
    dest = tmp_path / "4d"
    dest.mkdir()
    (tmp_path / "regions.json").write_text(
        json.dumps({"schema_version": "other", "regions": []}),
        encoding="utf-8",
    )
    assert load_region_boxes(dest) == []


def test_publisher_drops_rejected_events(tmp_path: Path) -> None:
    timeline = SceneTimeline(
        schema_version="ss-video.scene-timeline.v1",
        mission_id="mission-profile",
        profile="fast",
        coordinate_frame={"name": "mission_enu", "metric_scale": "unavailable"},
        model_manifest_ref="manifest.json",
        events=[
            {
                "event_id": "evt-keep",
                "type": "count_changed",
                "summary": "kept",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "participants": ["track-1"],
                "confidence": 0.9,
                "verification": Verification(status="accepted", rules=["count"], reasons=[]),
                "evidence": [
                    {
                        "frame_id": "mission-profile:track-1:0000000",
                        "t_sec": 0.0,
                        "track_ids": ["track-1"],
                    }
                ],
            },
            {
                "event_id": "evt-drop",
                "type": "count_changed",
                "summary": "dropped",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "participants": ["track-1"],
                "confidence": 0.2,
                "verification": Verification(status="rejected", rules=["count"], reasons=["no"]),
                "evidence": [],
            },
        ],
    )
    fresh = publish_accepted(
        tmp_path, timeline, zone_id="site", sensor_id="cam", published_at=_WHEN
    )
    assert fresh == ["evt-keep"]
    text = (tmp_path / "published-events.jsonl").read_text(encoding="utf-8")
    assert "evt-drop" not in text
    assert "evt-keep" in text
