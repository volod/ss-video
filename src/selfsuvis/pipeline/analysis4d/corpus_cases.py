"""Occlusion, crowded, and weak-calibration fixture missions."""

from pathlib import Path

from selfsuvis.pipeline.analysis4d.corpus import (
    _box,
    _degradation,
    _seal,
    _stage,
    _timeline,
    _track,
    _truth,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_GAP,
    CoordinateFrame,
    GapRecord,
    TruthCount,
    TruthDetection,
)


def _missing() -> list:
    return [_degradation("models_missing", "fixture corpus has no model inference")]


def write_occlusion(root: Path) -> None:
    """One track matches two ground-truth identities. The middle sample is occluded."""
    mission_id = "mission-occlusion"
    frame = CoordinateFrame(name="mission_enu", metric_scale="metric", calibration_id="cal-1")
    box_a = _box(0.10, 0.10, 0.20, 0.20)
    box_b = _box(0.60, 0.60, 0.20, 0.20)
    degradations = _missing()
    _seal(
        root / "occlusion-cut",
        mission_id=mission_id,
        profile="deep",
        frame=frame,
        degradations=degradations,
        stage=_stage(3),
        keyframes=[0.0, 9.0],
        timeline=_timeline(mission_id, "deep", frame, degradations),
        tracks=[
            _track(mission_id, "obs-a", "track-bad", "confirmed", 0.0, box_a),
            _track(mission_id, "obs-occ", "track-bad", "occluded", 5.0, box_a),
            _track(mission_id, "obs-b", "track-bad", "confirmed", 9.0, box_b),
        ],
        deltas=[],
        proposals=[],
        qa=[],
        gaps=[],
        geometry=[],
        masks=[],
        truth=_truth(
            mission_id,
            10.0,
            ["long_occlusion"],
            tracks=[
                TruthDetection(t_sec=0.0, gt_id="gt-1", xywh_norm=box_a.xywh_norm, label="crate"),
                TruthDetection(t_sec=9.0, gt_id="gt-2", xywh_norm=box_b.xywh_norm, label="crate"),
            ],
        ),
    )


def write_crowded(root: Path) -> None:
    """Two confirmed instances of one label."""
    mission_id = "mission-crowded"
    frame = CoordinateFrame(name="mission_enu", metric_scale="metric", calibration_id="cal-1")
    left = _box(0.10, 0.10, 0.20, 0.20)
    right = _box(0.60, 0.10, 0.20, 0.20)
    degradations = _missing()
    _seal(
        root / "crowded",
        mission_id=mission_id,
        profile="deep",
        frame=frame,
        degradations=degradations,
        stage=_stage(1),
        keyframes=[0.0],
        timeline=_timeline(mission_id, "deep", frame, degradations),
        tracks=[
            _track(mission_id, "obs-l", "track-l", "confirmed", 0.0, left),
            _track(mission_id, "obs-r", "track-r", "confirmed", 0.0, right),
        ],
        deltas=[],
        proposals=[],
        qa=[],
        gaps=[],
        geometry=[],
        masks=[],
        truth=_truth(
            mission_id,
            4.0,
            ["crowded_repeated_objects"],
            keyframe_boundaries_sec=[0.0],
            tracks=[
                TruthDetection(t_sec=0.0, gt_id="gt-l", xywh_norm=left.xywh_norm, label="crate"),
                TruthDetection(t_sec=0.0, gt_id="gt-r", xywh_norm=right.xywh_norm, label="crate"),
            ],
            counts=[TruthCount(t_sec=0.0, label="crate", count=2)],
        ),
    )


def write_weak(root: Path) -> None:
    """Unavailable scale, missing pose, and a gap for skipped frames."""
    mission_id = "mission-weak"
    frame = CoordinateFrame(name="mission_enu", metric_scale="unavailable", calibration_id=None)
    box = _box(0.20, 0.20, 0.20, 0.20)
    degradations = [
        *_missing(),
        _degradation("calibration_missing", "no calibration id"),
        _degradation("pose_missing", "no camera pose"),
    ]
    gap = GapRecord(
        schema_version=SCHEMA_GAP,
        mission_id=mission_id,
        gap_id="gap-1",
        start_sec=1.0,
        end_sec=2.0,
        reason="queue_coalesce",
        skipped_frames=2,
    )
    _seal(
        root / "weak-calibration",
        mission_id=mission_id,
        profile="fast",
        frame=frame,
        degradations=degradations,
        stage=_stage(
            1, skipped_frames=2, flags=["models_missing", "calibration_missing", "pose_missing"]
        ),
        keyframes=[],
        timeline=_timeline(mission_id, "fast", frame, degradations),
        tracks=[_track(mission_id, "obs-1", "track-1", "confirmed", 0.0, box)],
        deltas=[],
        proposals=[],
        qa=[],
        gaps=[gap],
        geometry=[],
        masks=[],
        truth=_truth(
            mission_id,
            5.0,
            ["weak_calibration"],
            tracks=[
                TruthDetection(t_sec=0.0, gt_id="gt-1", xywh_norm=box.xywh_norm, label="crate")
            ],
        ),
    )
