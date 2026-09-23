"""Relative-depth, contradiction, empty, and invalid fixture missions."""

from pathlib import Path

from selfsuvis.pipeline.analysis4d.corpus import (
    _box,
    _degradation,
    _edge,
    _event,
    _geometry,
    _node,
    _proposal,
    _seal,
    _stage,
    _timeline,
    _track,
    _truth,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    CoordinateFrame,
    TruthDepth,
    TruthDetection,
    TruthRelation,
    TruthVerifierLabel,
)


def _missing() -> list:
    return [_degradation("models_missing", "fixture corpus has no model inference")]


def write_relative(root: Path) -> None:
    """Relative depth only. Metric predicates stay off."""
    mission_id = "mission-relative"
    frame_name = "mission_enu"
    frame = CoordinateFrame(name=frame_name, metric_scale="relative", calibration_id=None)
    degradations = [
        *_missing(),
        _degradation("relative_depth_only", "depth is scale-free"),
        _degradation("metric_scale_missing", "metric predicates are disabled"),
    ]
    geometry = [
        (
            "geometry/n-track-1/0.json",
            _geometry(
                mission_id,
                "g-1",
                "n-track-1",
                0.0,
                frame_name,
                [1.0, 0.0, 0.5],
                [0.4, 0.4, 0.4],
                depth_m=1.0,
            ),
        ),
        (
            "geometry/n-track-1/2.json",
            _geometry(
                mission_id,
                "g-2",
                "n-track-1",
                2.0,
                frame_name,
                [1.2, 0.0, 0.5],
                [0.4, 0.4, 0.4],
                depth_m=2.0,
            ),
        ),
    ]
    _seal(
        root / "relative-depth",
        mission_id=mission_id,
        profile="deep",
        frame=frame,
        degradations=degradations,
        stage=_stage(2, flags=["models_missing", "relative_depth_only"]),
        keyframes=[0.0, 2.0],
        timeline=_timeline(mission_id, "deep", frame, degradations),
        tracks=[_track(mission_id, "obs-1", "track-1", "confirmed", 0.0, _box(0.2, 0.2, 0.2, 0.2))],
        deltas=[_node(mission_id, "delta-track", "n-track-1", "track", "crate", "track-1", 0.0)],
        proposals=[],
        qa=[],
        gaps=[],
        geometry=geometry,
        masks=[],
        truth=_truth(
            mission_id,
            4.0,
            ["relative_depth"],
            tracks=[
                TruthDetection(
                    t_sec=0.0,
                    gt_id="gt-1",
                    xywh_norm=[0.2, 0.2, 0.2, 0.2],
                    label="crate",
                )
            ],
            depths=[
                TruthDepth(t_sec=0.0, subject_id="n-track-1", depth_m=1.0),
                TruthDepth(t_sec=2.0, subject_id="n-track-1", depth_m=2.0),
            ],
        ),
    )


def write_contradiction(root: Path) -> None:
    """Accepted left-of edge plus a rejected right-of proposal."""
    mission_id = "mission-contradiction"
    frame_name = "mission_enu"
    frame = CoordinateFrame(name=frame_name, metric_scale="metric", calibration_id="cal-1")
    degradations = _missing()
    _seal(
        root / "contradiction",
        mission_id=mission_id,
        profile="deep",
        frame=frame,
        degradations=degradations,
        stage=_stage(1),
        keyframes=[0.0],
        timeline=_timeline(mission_id, "deep", frame, degradations),
        tracks=[
            _track(mission_id, "obs-a", "track-a", "confirmed", 0.0, _box(0.1, 0.1, 0.2, 0.2)),
            _track(mission_id, "obs-b", "track-b", "confirmed", 0.0, _box(0.6, 0.1, 0.2, 0.2)),
        ],
        deltas=[
            _node(mission_id, "delta-a", "n-a", "track", "crate", "track-a", 0.0),
            _node(mission_id, "delta-b", "n-b", "track", "crate", "track-b", 0.0),
            _edge(
                mission_id,
                "delta-left",
                "edge-left",
                "n-a",
                "left_of",
                "n-b",
                0.0,
                2.0,
                frame_name,
                "g-a",
            ),
        ],
        proposals=[
            _proposal(
                mission_id,
                "proposal-false",
                "relation",
                "n-a is right of n-b",
                "rejected",
                predicate="right_of",
            ),
            _proposal(mission_id, "proposal-true", "attribute", "two crates", "accepted"),
        ],
        qa=[],
        gaps=[],
        geometry=[
            (
                "geometry/n-a/0.json",
                _geometry(
                    mission_id, "g-a", "n-a", 0.0, frame_name, [0.0, 0.0, 0.5], [0.4, 0.4, 0.4]
                ),
            ),
            (
                "geometry/n-b/0.json",
                _geometry(
                    mission_id, "g-b", "n-b", 0.0, frame_name, [2.0, 0.0, 0.5], [0.4, 0.4, 0.4]
                ),
            ),
        ],
        masks=[],
        truth=_truth(
            mission_id,
            4.0,
            ["contradictory_proposal"],
            tracks=[
                TruthDetection(
                    t_sec=0.0, gt_id="gt-a", xywh_norm=[0.1, 0.1, 0.2, 0.2], label="crate"
                ),
                TruthDetection(
                    t_sec=0.0, gt_id="gt-b", xywh_norm=[0.6, 0.1, 0.2, 0.2], label="crate"
                ),
            ],
            relations=[
                TruthRelation(
                    subject_id="n-a",
                    predicate="left_of",
                    object_id="n-b",
                    start_sec=0.0,
                    end_sec=2.0,
                )
            ],
            verifier=[
                TruthVerifierLabel(proposal_id="proposal-false", should_accept=False),
                TruthVerifierLabel(proposal_id="proposal-true", should_accept=True),
            ],
        ),
    )


def write_empty(root: Path) -> None:
    """Empty verified timeline and zero QA pairs."""
    mission_id = "mission-empty"
    frame = CoordinateFrame(name="mission_enu", metric_scale="unavailable", calibration_id=None)
    degradations = [
        _degradation("empty_scene", "no claim passed verification"),
        *_missing(),
    ]
    _seal(
        root / "empty-scene",
        mission_id=mission_id,
        profile="fast",
        frame=frame,
        degradations=degradations,
        stage=_stage(0, flags=["empty_scene", "models_missing"]),
        keyframes=[],
        timeline=_timeline(mission_id, "fast", frame, degradations),
        tracks=[],
        deltas=[],
        proposals=[],
        qa=[],
        gaps=[],
        geometry=[],
        masks=[],
        truth=_truth(mission_id, 5.0, ["empty_scene"]),
    )


def write_invalid_bundles(root: Path) -> None:
    """Bundles that must fail validation: a dangling id and a false accepted edge."""
    _dangling(root)
    _bad_edge(root)


def _dangling(root: Path) -> None:
    mission_id = "mission-dangling"
    frame = CoordinateFrame(name="mission_enu", metric_scale="metric", calibration_id="cal-1")
    event = _event(
        "evt-1",
        "missing track entered",
        ["track-missing"],
        "delta-missing",
        "mission-dangling:f1:1000",
        1.0,
        "track-1",
    )
    degradations = _missing()
    _seal(
        root / "invalid" / "dangling",
        mission_id=mission_id,
        profile="fast",
        frame=frame,
        degradations=degradations,
        stage=_stage(1),
        keyframes=[],
        timeline=_timeline(mission_id, "fast", frame, degradations, [event]),
        tracks=[_track(mission_id, "obs-1", "track-1", "confirmed", 1.0, _box(0.1, 0.1, 0.2, 0.2))],
        deltas=[],
        proposals=[],
        qa=[],
        gaps=[],
        geometry=[],
        masks=[],
        truth=_truth(mission_id, 3.0, ["dangling"]),
    )


def _bad_edge(root: Path) -> None:
    mission_id = "mission-bad-edge"
    frame_name = "mission_enu"
    frame = CoordinateFrame(name=frame_name, metric_scale="metric", calibration_id="cal-1")
    degradations = _missing()
    _seal(
        root / "invalid" / "bad-edge",
        mission_id=mission_id,
        profile="deep",
        frame=frame,
        degradations=degradations,
        stage=_stage(1),
        keyframes=[],
        timeline=_timeline(mission_id, "deep", frame, degradations),
        tracks=[
            _track(mission_id, "obs-a", "track-a", "confirmed", 0.0, _box(0.1, 0.1, 0.2, 0.2)),
            _track(mission_id, "obs-b", "track-b", "confirmed", 0.0, _box(0.6, 0.1, 0.2, 0.2)),
        ],
        deltas=[
            _node(mission_id, "delta-a", "n-a", "track", "crate", "track-a", 0.0),
            _node(mission_id, "delta-b", "n-b", "track", "crate", "track-b", 0.0),
            _edge(
                mission_id,
                "delta-right",
                "edge-right",
                "n-a",
                "right_of",
                "n-b",
                0.0,
                2.0,
                frame_name,
                "g-a",
            ),
        ],
        proposals=[],
        qa=[],
        gaps=[],
        geometry=[
            (
                "geometry/n-a/0.json",
                _geometry(
                    mission_id, "g-a", "n-a", 0.0, frame_name, [0.0, 0.0, 0.5], [0.4, 0.4, 0.4]
                ),
            ),
            (
                "geometry/n-b/0.json",
                _geometry(
                    mission_id, "g-b", "n-b", 0.0, frame_name, [2.0, 0.0, 0.5], [0.4, 0.4, 0.4]
                ),
            ),
        ],
        masks=[],
        truth=_truth(mission_id, 3.0, ["bad_edge"]),
    )
