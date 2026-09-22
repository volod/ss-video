"""Fast and deep profile orchestration."""

import json
from datetime import UTC, datetime
from pathlib import Path

from selfsuvis.pipeline.analysis4d.budget import BoundedFrameQueue, BudgetController
from selfsuvis.pipeline.analysis4d.io import read_jsonl, read_model
from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal
from selfsuvis.pipeline.analysis4d.profile import scheduled_analysis_jobs
from selfsuvis.pipeline.analysis4d.providers import Prompt, ScriptedGrounding
from selfsuvis.pipeline.analysis4d.publish import publish_accepted
from selfsuvis.pipeline.analysis4d.schemas import (
    AnalysisManifest,
    GapRecord,
    SceneTimeline,
    Verification,
)
from selfsuvis.pipeline.analysis4d.tracks import Detection
from selfsuvis.pipeline.analysis4d.verified_scene_event import VerifiedSceneEvent
from selfsuvis.pipeline.workflows.analysis4d_profile import run_deep_profile, run_fast_profile

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
