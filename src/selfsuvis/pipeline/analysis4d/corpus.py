"""Deterministic 4D fixture corpus.

Regenerate with ``build_corpus(path)``. ``invalid/`` is not scored.
"""

import json
import shutil
from pathlib import Path

from selfsuvis.pipeline.analysis4d.io import (
    canonical_bytes,
    canonical_line,
    sha256_bytes,
    write_bytes,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_DELTA,
    SCHEMA_GAP,
    SCHEMA_GEOMETRY,
    SCHEMA_MANIFEST,
    SCHEMA_MASK,
    SCHEMA_PROPOSAL,
    SCHEMA_QA,
    SCHEMA_TIMELINE,
    SCHEMA_TRACK,
    SCHEMA_TRUTH,
    AnalysisManifest,
    ArtifactEntry,
    Box2D,
    Degradation,
    EvidenceRef,
    GeometrySample,
    GraphDelta,
    GraphEdge,
    GraphNode,
    IdentityLink,
    Location3D,
    MissionTruth,
    Proposal,
    QaAnswer,
    QaPair,
    QaRecord,
    SceneTimeline,
    StageReport,
    TimelineEvent,
    TrackRecord,
    Verification,
)

CORPUS_ID = "analysis4d-v1"
REQUIRED_CONDITIONS = (
    "camera_cut",
    "long_occlusion",
    "crowded_repeated_objects",
    "moving_camera",
    "weak_calibration",
    "relative_depth",
    "contradictory_proposal",
    "empty_scene",
)
FIXTURE_TIME = "2026-01-15T00:00:00Z"
_QUAT = [0.0, 0.0, 0.0, 1.0]
MISSIONS = (
    "nominal",
    "occlusion-cut",
    "crowded",
    "weak-calibration",
    "relative-depth",
    "contradiction",
    "empty-scene",
)


def mask_bits(size: int = 8) -> str:
    """Row-major mask with a one-pixel background border."""
    chars = []
    for row in range(size):
        for col in range(size):
            border = row in {0, size - 1} or col in {0, size - 1}
            chars.append("0" if border else "1")
    return "".join(chars)


def build_corpus(root: Path) -> None:
    """Write the pinned corpus, replacing ``root`` when it already exists."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    from selfsuvis.pipeline.analysis4d.corpus_cases import (
        write_crowded,
        write_occlusion,
        write_weak,
    )
    from selfsuvis.pipeline.analysis4d.corpus_nominal import write_nominal
    from selfsuvis.pipeline.analysis4d.corpus_rest import (
        write_contradiction,
        write_empty,
        write_invalid_bundles,
        write_relative,
    )

    write_nominal(root)
    write_occlusion(root)
    write_crowded(root)
    write_weak(root)
    write_relative(root)
    write_contradiction(root)
    write_empty(root)
    _invalid_documents(root)
    write_invalid_bundles(root)
    index = {
        "corpus_id": CORPUS_ID,
        "missions": list(MISSIONS),
        "required_conditions": list(REQUIRED_CONDITIONS),
    }
    write_bytes(
        root / "corpus.json", (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _box(x: float, y: float, width: float, height: float) -> Box2D:
    return Box2D(xywh_norm=[x, y, width, height])


def _degradation(code: str, detail: str) -> Degradation:
    return Degradation(code=code, stage="contract", detail=detail)


def _stage(
    processed_frames: int, skipped_frames: int = 0, flags: list[str] | None = None
) -> StageReport:
    return StageReport(
        stage="contract",
        queue_delay_sec=0.0,
        inference_time_sec=0.0,
        processed_frames=processed_frames,
        skipped_frames=skipped_frames,
        trigger_reason="fixture",
        degradation_flags=flags or ["models_missing"],
        queue_depth=0,
    )


def _track(
    mission_id, observation_id, track_id, state, t_sec, box, label="crate", mask_ref=None, link=None
):
    return TrackRecord(
        schema_version=SCHEMA_TRACK,
        mission_id=mission_id,
        observation_id=observation_id,
        track_id=track_id,
        state=state,
        t_sec=t_sec,
        prompt_id="prompt-crate",
        label_raw=label,
        label_normalized=label,
        box=box,
        mask_ref=mask_ref,
        confidence=0.9,
        identity_link=None
        if link is None
        else IdentityLink(linked_track_id=link, reason="camera_cut"),
    )


def _node(mission_id, delta_id, node_id, kind, label, track_id, t_sec) -> GraphDelta:
    return GraphDelta(
        schema_version=SCHEMA_DELTA,
        mission_id=mission_id,
        delta_id=delta_id,
        op="add",
        t_sec=t_sec,
        node=GraphNode(node_id=node_id, kind=kind, label=label, track_id=track_id),
    )


def _edge(
    mission_id, delta_id, edge_id, subject_id, predicate, object_id, start, end, frame, evidence_id
) -> GraphDelta:
    return GraphDelta(
        schema_version=SCHEMA_DELTA,
        mission_id=mission_id,
        delta_id=delta_id,
        op="add",
        t_sec=start,
        edge=GraphEdge(
            edge_id=edge_id,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            start_sec=start,
            end_sec=end,
            coordinate_frame=frame,
            confidence=0.95,
            source="deterministic",
            evidence_ids=[evidence_id],
            verification_status="accepted",
        ),
    )


def _geometry(
    mission_id, sample_id, subject_id, t_sec, frame, center, extent, depth_m=None, pose=None
):
    return GeometrySample(
        schema_version=SCHEMA_GEOMETRY,
        mission_id=mission_id,
        sample_id=sample_id,
        subject_id=subject_id,
        t_sec=t_sec,
        frame=frame,
        center_m=center,
        extent_m=extent,
        quaternion_xyzw=list(_QUAT),
        depth_m=depth_m,
        pose_position_m=pose,
    )


def _event(
    event_id,
    summary,
    participants,
    delta_id,
    frame_id,
    t_sec,
    track_id,
    frame=None,
    center=None,
    geometry_ref=None,
):
    location = None
    if center is not None and frame is not None:
        location = Location3D(frame=frame, center_m=center, covariance_diag=[0.1, 0.1, 0.1])
    return TimelineEvent(
        event_id=event_id,
        type="entered_region",
        summary=summary,
        start_sec=4.0 if center is not None else 1.0,
        end_sec=6.0 if center is not None else 2.0,
        participants=participants,
        state_delta_refs=[delta_id] if delta_id else [],
        location=location,
        confidence=0.94,
        verification=Verification(status="accepted", rules=["geometry"], reasons=[]),
        evidence=[
            EvidenceRef(
                frame_id=frame_id,
                t_sec=t_sec,
                track_ids=[track_id],
                geometry_ref=geometry_ref,
            )
        ],
    )


def _qa_pair(mission_id: str) -> QaRecord:
    return QaRecord(
        schema_version=SCHEMA_QA,
        mission_id=mission_id,
        qa_id="qa-1",
        type="spatial",
        question="Which tracked object entered the zone?",
        answer=QaAnswer(kind="track_ref", value="track-1", unit=None),
        graph_program="entered(?track, n-zone, after=4.0)",
        interval_sec=[4.0, 6.0],
        evidence_event_ids=["evt-1"],
        verification_status="accepted",
    )


def _proposal(mission_id, proposal_id, kind, text, status, predicate=None) -> Proposal:
    return Proposal(
        schema_version=SCHEMA_PROPOSAL,
        mission_id=mission_id,
        proposal_id=proposal_id,
        claim_kind=kind,
        text=text,
        predicate=predicate,
        verification_status=status,
        reasons=[] if status == "accepted" else ["geometry_disagrees"],
    )


def _timeline(mission_id, profile, frame, degradations, events=None, qa=None) -> SceneTimeline:
    pairs = []
    for record in qa or []:
        data = record.model_dump()
        data.pop("schema_version")
        data.pop("mission_id")
        pairs.append(QaPair(**data))
    return SceneTimeline(
        schema_version=SCHEMA_TIMELINE,
        mission_id=mission_id,
        profile=profile,
        coordinate_frame=frame,
        model_manifest_ref="manifest.json",
        degradations=degradations,
        events=events or [],
        qa_pairs=pairs,
    )


def _truth(mission_id, duration, conditions, **kwargs) -> MissionTruth:
    return MissionTruth(
        schema_version=SCHEMA_TRUTH,
        mission_id=mission_id,
        media_duration_sec=duration,
        conditions=conditions,
        **kwargs,
    )


def _seal(
    dest: Path,
    *,
    mission_id,
    profile,
    frame,
    degradations,
    stage,
    keyframes,
    timeline,
    tracks,
    deltas,
    proposals,
    qa,
    gaps,
    geometry,
    masks,
    truth,
) -> None:
    """Write one mission directory and its manifest."""
    files: list[tuple[str, str, str, bytes]] = []

    def add(path: str, kind: str, schema: str, payload: bytes) -> None:
        write_bytes(dest / path, payload)
        files.append((path, kind, schema, payload))

    add("tracks.jsonl", "tracks", SCHEMA_TRACK, b"".join(canonical_line(row) for row in tracks))
    add(
        "graph-deltas.jsonl",
        "graph_deltas",
        SCHEMA_DELTA,
        b"".join(canonical_line(row) for row in deltas),
    )
    add(
        "proposals.jsonl",
        "proposals",
        SCHEMA_PROPOSAL,
        b"".join(canonical_line(row) for row in proposals),
    )
    add("timeline.json", "timeline", SCHEMA_TIMELINE, canonical_bytes(timeline))
    add("qa.jsonl", "qa", SCHEMA_QA, b"".join(canonical_line(row) for row in qa))
    if gaps:
        add("gaps.jsonl", "gaps", SCHEMA_GAP, b"".join(canonical_line(row) for row in gaps))
    for path, sample in geometry:
        add(path, "geometry", SCHEMA_GEOMETRY, canonical_bytes(sample))
    for path, mask in masks:
        add(path, "masks", SCHEMA_MASK, canonical_bytes(mask))
    write_bytes(dest / "truth.json", canonical_bytes(truth))
    manifest = AnalysisManifest(
        schema_version=SCHEMA_MANIFEST,
        mission_id=mission_id,
        profile=profile,
        coordinate_frame=frame,
        created_at=FIXTURE_TIME,
        artifacts=[
            ArtifactEntry(
                path=path,
                kind=kind,
                schema_version=schema,
                sha256=sha256_bytes(payload),
                bytes=len(payload),
            )
            for path, kind, schema, payload in files
        ],
        models=[],
        degradations=degradations,
        stages=[stage],
        keyframes_sec=keyframes,
    )
    write_bytes(dest / "manifest.json", canonical_bytes(manifest))


def _invalid_documents(root: Path) -> None:
    dest = root / "invalid"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "unversioned.json").write_text(
        json.dumps({"mission_id": "mission-x", "profile": "fast", "events": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    interval = {
        "schema_version": SCHEMA_TIMELINE,
        "mission_id": "mission-x",
        "profile": "fast",
        "coordinate_frame": {
            "name": "mission_enu",
            "metric_scale": "unavailable",
            "calibration_id": None,
        },
        "model_manifest_ref": "manifest.json",
        "degradations": [],
        "events": [
            {
                "event_id": "evt-1",
                "type": "entered_region",
                "summary": "bad interval",
                "start_sec": 5.0,
                "end_sec": 4.0,
                "participants": ["track-1"],
                "state_delta_refs": [],
                "location": None,
                "confidence": 0.5,
                "verification": {
                    "status": "uncertain",
                    "rules": [],
                    "vlm_claim_ref": None,
                    "reasons": [],
                },
                "evidence": [],
                "supersedes": None,
            }
        ],
        "qa_pairs": [],
    }
    mixed = json.loads(json.dumps(interval))
    mixed["coordinate_frame"] = {
        "name": "mission_enu",
        "metric_scale": "metric",
        "calibration_id": "cal-1",
    }
    mixed["events"][0]["start_sec"] = 1.0
    mixed["events"][0]["end_sec"] = 2.0
    mixed["events"][0]["summary"] = "mixed frame"
    mixed["events"][0]["location"] = {
        "frame": "camera_0",
        "center_m": [1.0, 2.0, 3.0],
        "covariance_diag": [0.1, 0.1, 0.1],
    }
    mixed["events"][0]["evidence"] = [
        {
            "frame_id": "mission-x:f1:1000",
            "t_sec": 1.0,
            "track_ids": ["track-1"],
            "mask_ref": None,
            "geometry_ref": None,
        }
    ]
    (dest / "invalid-interval.json").write_text(
        json.dumps(interval, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "mixed-frame.json").write_text(json.dumps(mixed, indent=2) + "\n", encoding="utf-8")
