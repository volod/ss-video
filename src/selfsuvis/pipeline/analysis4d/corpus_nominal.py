"""Nominal metric mission: camera cut, moving camera, and a verified event."""

from pathlib import Path

from selfsuvis.pipeline.analysis4d.corpus import (
    _box,
    _degradation,
    _edge,
    _event,
    _geometry,
    _node,
    _proposal,
    _qa_pair,
    _seal,
    _stage,
    _timeline,
    _track,
    _truth,
    mask_bits,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_MASK,
    CoordinateFrame,
    MaskArtifact,
    TruthBox,
    TruthCount,
    TruthDepth,
    TruthDetection,
    TruthEvent,
    TruthMask,
    TruthQa,
    TruthRelation,
    TruthVerifierLabel,
)


def write_nominal(root: Path) -> None:
    """Write the nominal mission."""
    mission_id = "mission-nominal"
    frame_name = "mission_enu"
    frame = CoordinateFrame(name=frame_name, metric_scale="metric", calibration_id="cal-1")
    bits = mask_bits()
    mask_path = "masks/track-1/0.json"
    geom_track_4 = "geometry/n-track-1/4.json"
    box0 = _box(0.10, 0.20, 0.20, 0.30)
    box4 = _box(0.40, 0.20, 0.20, 0.30)
    box8 = _box(0.42, 0.22, 0.20, 0.30)
    tracks = [
        _track(mission_id, "obs-1", "track-1", "confirmed", 0.0, box0, mask_ref=mask_path),
        _track(mission_id, "obs-2", "track-1", "confirmed", 4.0, box4),
        _track(mission_id, "obs-3", "track-1", "ended", 8.0, box4),
        _track(mission_id, "obs-4", "track-3", "confirmed", 8.0, box8, link="track-1"),
    ]
    deltas = [
        _node(mission_id, "delta-zone", "n-zone", "region", "loading-zone", None, 0.0),
        _node(mission_id, "delta-track", "n-track-1", "track", "crate", "track-1", 0.0),
        _node(mission_id, "delta-track-3", "n-track-3", "track", "crate", "track-3", 8.0),
        _edge(
            mission_id,
            "delta-contains",
            "edge-contains",
            "n-zone",
            "contains",
            "n-track-1",
            4.0,
            10.0,
            frame_name,
            "g-track-4",
        ),
    ]
    geometry = [
        (
            "geometry/n-zone/4.json",
            _geometry(
                mission_id,
                "g-zone-4",
                "n-zone",
                4.0,
                frame_name,
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 4.0],
            ),
        ),
        (
            "geometry/n-track-1/0.json",
            _geometry(
                mission_id,
                "g-track-0",
                "n-track-1",
                0.0,
                frame_name,
                [6.0, 0.0, 0.5],
                [0.4, 0.4, 0.4],
                depth_m=6.0,
                pose=[0.0, 0.0, 1.5],
            ),
        ),
        (
            geom_track_4,
            _geometry(
                mission_id,
                "g-track-4",
                "n-track-1",
                4.0,
                frame_name,
                [1.0, 0.0, 0.5],
                [0.4, 0.4, 0.4],
                depth_m=4.0,
                pose=[0.4, 0.2, 1.5],
            ),
        ),
        (
            "geometry/n-track-3/8.json",
            _geometry(
                mission_id,
                "g-track-8",
                "n-track-3",
                8.0,
                frame_name,
                [1.1, 0.1, 0.5],
                [0.4, 0.4, 0.4],
                depth_m=3.8,
                pose=[0.6, 0.3, 1.5],
            ),
        ),
    ]
    masks = [
        (
            mask_path,
            MaskArtifact(
                schema_version=SCHEMA_MASK,
                mission_id=mission_id,
                track_id="track-1",
                t_sec=0.0,
                width=8,
                height=8,
                bits=bits,
            ),
        )
    ]
    event = _event(
        "evt-1",
        "track-1 entered loading-zone",
        ["track-1", "n-zone"],
        "delta-contains",
        "mission-nominal:f4:4000",
        4.0,
        "track-1",
        frame=frame_name,
        center=[1.0, 0.0, 0.5],
        geometry_ref=geom_track_4,
    )
    qa = [_qa_pair(mission_id)]
    proposals = [
        _proposal(mission_id, "proposal-ok", "attribute", "crate is a crate", "accepted"),
        _proposal(
            mission_id,
            "proposal-bad",
            "relation",
            "track-1 is right of the zone",
            "rejected",
            predicate="right_of",
        ),
    ]
    degradations = [_degradation("models_missing", "fixture corpus has no model inference")]
    truth = _truth(
        mission_id,
        10.0,
        ["moving_camera", "camera_cut"],
        keyframe_boundaries_sec=[4.0],
        tracks=[
            TruthDetection(t_sec=0.0, gt_id="gt-1", xywh_norm=box0.xywh_norm, label="crate"),
            TruthDetection(t_sec=4.0, gt_id="gt-1", xywh_norm=box4.xywh_norm, label="crate"),
            TruthDetection(t_sec=8.0, gt_id="gt-1", xywh_norm=box8.xywh_norm, label="crate"),
        ],
        counts=[
            TruthCount(t_sec=0.0, label="crate", count=1),
            TruthCount(t_sec=4.0, label="crate", count=1),
        ],
        masks=[TruthMask(t_sec=0.0, track_id="track-1", width=8, height=8, bits=bits)],
        depths=[
            TruthDepth(t_sec=0.0, subject_id="n-track-1", depth_m=6.0),
            TruthDepth(t_sec=4.0, subject_id="n-track-1", depth_m=4.0),
        ],
        boxes=[
            TruthBox(
                t_sec=0.0,
                subject_id="n-track-1",
                center_m=[6.0, 0.0, 0.5],
                extent_m=[0.4, 0.4, 0.4],
            ),
            TruthBox(
                t_sec=4.0,
                subject_id="n-track-1",
                center_m=[1.0, 0.0, 0.5],
                extent_m=[0.4, 0.4, 0.4],
            ),
        ],
        relations=[
            TruthRelation(
                subject_id="n-zone",
                predicate="contains",
                object_id="n-track-1",
                start_sec=4.0,
                end_sec=10.0,
            )
        ],
        events=[
            TruthEvent(
                type="entered_region",
                start_sec=4.0,
                end_sec=6.0,
                participants=["track-1", "n-zone"],
            )
        ],
        qa=[TruthQa(qa_id="qa-1", answer_value="track-1")],
        verifier=[
            TruthVerifierLabel(proposal_id="proposal-ok", should_accept=True),
            TruthVerifierLabel(proposal_id="proposal-bad", should_accept=False),
        ],
    )
    _seal(
        root / "nominal",
        mission_id=mission_id,
        profile="deep",
        frame=frame,
        degradations=degradations,
        stage=_stage(3),
        keyframes=[0.0, 4.0, 8.0],
        timeline=_timeline(mission_id, "deep", frame, degradations, [event], qa),
        tracks=tracks,
        deltas=deltas,
        proposals=proposals,
        qa=qa,
        gaps=[],
        geometry=geometry,
        masks=masks,
        truth=truth,
    )
