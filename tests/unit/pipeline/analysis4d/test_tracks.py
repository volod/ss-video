"""Keyframe selector, track lifecycle, and the fixture half of the track benchmark."""

from pathlib import Path

from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal, SelectorConfig, select_keyframes
from selfsuvis.pipeline.analysis4d.passes import write_profiles
from selfsuvis.pipeline.analysis4d.pin import ProviderProbe, choose_pin
from selfsuvis.pipeline.analysis4d.providers import (
    CountHit,
    GroundingDinoProvider,
    Prompt,
    ScriptedCount,
    ScriptedGrounding,
)
from selfsuvis.pipeline.analysis4d.track_benchmark import run_track_benchmark
from selfsuvis.pipeline.analysis4d.tracks import Detection, Tracker, TrackerConfig
from selfsuvis.pipeline.analysis4d.validate import validate_bundle

ASSETS = Path(__file__).resolve().parents[4] / "tests" / "assets" / "analysis4d"
_PROMPT = Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")


def _detection(xywh: tuple[float, float, float, float], **kwargs) -> Detection:
    values = {
        "prompt_id": "prompt-crate",
        "label_raw": "crate",
        "label_normalized": "crate",
        "xywh": xywh,
        "confidence": 0.9,
        "appearance": (1.0, 0.0),
    }
    values.update(kwargs)
    return Detection(**values)


def test_maxinfo_prefers_an_orthogonal_frame() -> None:
    frames = [
        FrameSignal(t_sec=0.0, embedding=(1.0, 0.0), drift=1.0),
        FrameSignal(t_sec=1.0, embedding=(1.0, 0.0), drift=1.0),
        FrameSignal(t_sec=2.0, embedding=(0.0, 1.0), drift=1.0),
        FrameSignal(t_sec=3.0, embedding=(0.0, 1.0), drift=1.0),
    ]
    config = SelectorConfig(
        chunk_sec=10.0, keyframe_budget=2, min_spacing_sec=0.0, max_gap_sec=100.0
    )
    picks, _skips = select_keyframes(frames, config, profile="fast")
    assert [pick.t_sec for pick in picks] == [0.0, 2.0]
    assert picks[0].forced
    assert "diversity" in picks[1].reasons


def test_same_seed_selects_the_same_frames() -> None:
    frames = [
        FrameSignal(t_sec=0.0, embedding=(1.0, 0.0), drift=1.0),
        FrameSignal(t_sec=0.0, embedding=(0.0, 1.0), drift=1.0),
    ]
    first, _ = select_keyframes(frames, SelectorConfig(seed=3, keyframe_budget=1), profile="fast")
    second, _ = select_keyframes(frames, SelectorConfig(seed=3, keyframe_budget=1), profile="fast")
    assert [(pick.t_sec, pick.reasons) for pick in first] == [
        (pick.t_sec, pick.reasons) for pick in second
    ]


def test_heartbeat_limits_the_gap() -> None:
    frames = [FrameSignal(t_sec=float(index), embedding=(1.0, 0.0)) for index in range(13)]
    picks, _skips = select_keyframes(
        frames, SelectorConfig(max_gap_sec=4.0, min_spacing_sec=0.0), profile="fast"
    )
    times = [pick.t_sec for pick in picks]
    assert times[0] == 0.0
    assert all(right - left <= 4.0 for left, right in zip(times, times[1:]))
    assert any(reason == "max_gap" for pick in picks for reason in pick.reasons)


def test_quality_skip_is_a_gap(tmp_path: Path) -> None:
    frames = [
        FrameSignal(t_sec=0.0, embedding=(1.0, 0.0), drift=1.0),
        FrameSignal(t_sec=1.0, quality_ok=False),
        FrameSignal(t_sec=2.0, embedding=(0.0, 1.0), drift=1.0),
    ]
    dest = write_profiles(
        "mission-gap",
        frames,
        [_PROMPT],
        tmp_path,
        grounding=ScriptedGrounding({}),
        profiles=("fast",),
        selector=SelectorConfig(max_gap_sec=30.0),
    )
    bundle = validate_bundle(dest)
    assert bundle.gaps
    assert bundle.gaps[0].reason == "quality"
    assert sum(stage.skipped_frames for stage in bundle.manifest.stages) == 1


def test_corpus_boundaries_are_covered() -> None:
    report = run_track_benchmark(ASSETS, probe_gpu=False)
    assert report.passed
    assert report.keyframe_event_coverage == 1
    assert report.monotonic
    assert report.seed_stable
    assert report.silent_id_reuse is False


def test_occlusion_bundle_is_monotonic_and_linked(tmp_path: Path) -> None:
    report = run_track_benchmark(ASSETS, tmp_path, probe_gpu=False, seed=7)
    assert report.passed, report.failures
    bundle = validate_bundle(tmp_path / "occlusion-cut")
    times = [row.t_sec for row in bundle.tracks]
    assert times == sorted(times)
    deep = [row for row in bundle.tracks if row.observation_id.startswith("obs-deep-")]
    fast = [row for row in bundle.tracks if row.observation_id.startswith("obs-fast-")]
    deep_ids = {row.track_id for row in deep}
    fast_ids = {row.track_id for row in fast}
    assert len(deep_ids) >= 2
    assert len(fast_ids) >= 2
    assert any(row.identity_link is not None for row in deep)
    assert all(row.identity_link is None for row in fast)
    assert bundle.manifest.models == []
    assert any(note.reason == "scene_cut" for note in _audit_resets(bundle))


def test_count_disagreement_does_not_add_tracks(tmp_path: Path) -> None:
    frames = [FrameSignal(t_sec=0.0, embedding=(1.0, 0.0), drift=1.0)]
    left = _detection((0.10, 0.10, 0.20, 0.20))
    right = _detection((0.60, 0.10, 0.20, 0.20))
    counted = Detection(
        prompt_id="prompt-crate",
        label_raw="crate",
        label_normalized="crate",
        xywh=(0.40, 0.40, 0.10, 0.10),
        confidence=0.5,
        count_only=True,
    )
    dest = write_profiles(
        "mission-count",
        frames,
        [_PROMPT],
        tmp_path,
        grounding=ScriptedGrounding({0.0: [left, right, counted]}),
        counter=ScriptedCount({0.0: [CountHit("prompt-crate", "crate", 5)]}),
        profiles=("fast",),
        selector=SelectorConfig(max_gap_sec=30.0),
    )
    bundle = validate_bundle(dest)
    ids = {row.track_id for row in bundle.tracks}
    assert len(ids) == 2
    audit = _audit(bundle)
    assert audit.count_disagreements
    assert audit.count_disagreements[0].track_count == 2
    assert audit.count_disagreements[0].provider_count == 5
    assert "count_disagreement" in audit.keyframes[0].reasons
    assert any("count_disagreement" in stage.trigger_reason for stage in bundle.manifest.stages)


def test_failed_grounding_records_provider_unavailable(tmp_path: Path) -> None:
    provider = GroundingDinoProvider()
    provider.failed = True
    dest = write_profiles(
        "mission-down",
        [FrameSignal(t_sec=0.0, embedding=(1.0, 0.0))],
        [_PROMPT],
        tmp_path,
        grounding=provider,
        profiles=("fast",),
        selector=SelectorConfig(max_gap_sec=30.0),
    )
    bundle = validate_bundle(dest)
    assert bundle.tracks == []
    codes = {item.code for item in bundle.manifest.degradations}
    assert "provider_unavailable" in codes


def test_long_occlusion_does_not_reuse_an_id() -> None:
    tracker = Tracker(mission_id="mission-occlusion", profile="fast", config=TrackerConfig())
    frames = [
        FrameSignal(t_sec=0.0),
        FrameSignal(t_sec=1.0),
        FrameSignal(t_sec=5.0),
        FrameSignal(t_sec=9.0),
    ]
    boxes = {
        0.0: [_detection((0.10, 0.10, 0.20, 0.20))],
        1.0: [_detection((0.12, 0.10, 0.20, 0.20))],
        9.0: [_detection((0.60, 0.60, 0.20, 0.20))],
    }
    for frame in frames:
        tracker.step(frame, boxes.get(frame.t_sec, []))
    ids = {row.track_id for row in tracker.observations}
    assert len(ids) == 2
    assert all(row.identity_link_id is None for row in tracker.observations)
    times = [row.t_sec for row in tracker.observations]
    assert times == sorted(times)


def test_choose_pin_keeps_a_single_pair_and_rejects_counts() -> None:
    probes = [
        ProviderProbe(
            "grounding_dino", "grounding", "Apache-2.0", "allowed", True, 0.2, 1000, "ok"
        ),
        ProviderProbe("yolo_sam", "grounding", "AGPL-3.0", "allowed", True, 0.1, 1000, "ok"),
        ProviderProbe("sam2", "mask", "Apache-2.0", "allowed", True, 0.2, 1000, "ok"),
        ProviderProbe("sam3", "joint", "SAM-License", "unset", True, 0.2, 1000, "license"),
        ProviderProbe("countgd", "count", "unset", "unset", False, None, None, "missing"),
        ProviderProbe("kinematic", "mask", "none", "allowed", True, 0.0, 0, "box"),
    ]
    choice = choose_pin(probes)
    assert choice.model_gate == "pass"
    assert choice.grounding == "grounding_dino"
    assert choice.mask == "sam2"
    assert choice.count == ""
    blocked = choose_pin(
        [ProviderProbe("yolo_sam", "grounding", "AGPL-3.0", "rejected", True, 0.1, 1, "no")]
    )
    assert blocked.model_gate == "no-go"
    assert blocked.count == ""


def _audit(bundle):
    from selfsuvis.pipeline.analysis4d.io import read_model
    from selfsuvis.pipeline.analysis4d.schemas import TrackAudit

    return read_model(bundle.root / "track-audit.json", TrackAudit)


def _audit_resets(bundle):
    return _audit(bundle).memory_resets
