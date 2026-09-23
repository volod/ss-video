"""Read APIs for a mission's temporal scene graph, timeline, and Video-QA.

These routes expose graph deltas, the materialized view, proposals, the
verified timeline, and evidence cited by an event or a QA answer.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.pipeline.analysis4d.graph import materialize
from selfsuvis.pipeline.analysis4d.io import read_jsonl, read_model
from selfsuvis.pipeline.analysis4d.paths import analysis_dir
from selfsuvis.pipeline.analysis4d.qa import publishable_qa
from selfsuvis.pipeline.analysis4d.schemas import (
    GeometrySample,
    GraphDelta,
    MaskArtifact,
    Proposal,
    QaRecord,
    SceneTimeline,
)
from selfsuvis.pipeline.analysis4d.verifier import publishable_events
from selfsuvis.pipeline.workflows.analysis4d_profile import read_profile_status

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)


@router.get("/{mission_id}/4d/graph")
def read_scene_graph(
    mission_id: str,
    t_sec: float | None = Query(default=None, ge=0),
) -> dict:
    """Return the materialized scene graph.

    ``t_sec`` keeps edges whose half-open interval contains that time.
    Omitted, the response includes every non-superseded interval.
    """
    deltas = _deltas(mission_id)
    view = materialize(deltas)
    edges = view.edges
    if t_sec is not None:
        edges = [edge for edge in edges if edge.start_sec <= t_sec < edge.end_sec]
    return {
        "mission_id": mission_id,
        "nodes": [node.model_dump(mode="json") for node in view.nodes],
        "superseded_nodes": [node.model_dump(mode="json") for node in view.superseded_nodes],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "superseded_delta_ids": view.superseded_delta_ids,
    }


@router.get("/{mission_id}/4d/deltas")
def read_scene_deltas(mission_id: str) -> dict:
    """Return the append-only graph log in replay order."""
    deltas = _deltas(mission_id)
    ordered = sorted(deltas, key=lambda row: (row.t_sec, row.delta_id))
    return {
        "mission_id": mission_id,
        "deltas": [row.model_dump(mode="json") for row in ordered],
    }


@router.get("/{mission_id}/4d/proposals")
def read_scene_proposals(mission_id: str) -> dict:
    """Return proposed, corrected, rejected, and superseded claims."""
    root = _root(mission_id)
    proposals = read_jsonl(root / "proposals.jsonl", Proposal)
    return {
        "mission_id": mission_id,
        "proposals": [row.model_dump(mode="json") for row in proposals],
    }


@router.get("/{mission_id}/4d/status")
def read_analysis_status(mission_id: str) -> dict:
    """Return profile, queue depth, degradations, backlog, and published events."""
    try:
        return read_profile_status(mission_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="analysis profile not found") from exc


@router.get("/{mission_id}/4d/timeline")
def read_timeline(mission_id: str) -> dict:
    """Return the verified timeline, including rejected and uncertain events."""
    root = _root(mission_id)
    timeline = read_model(root / "timeline.json", SceneTimeline)
    return timeline.model_dump(mode="json")


@router.get("/{mission_id}/4d/qa")
def read_qa(mission_id: str) -> dict:
    """Return Video-QA rows. Accepted answers cite evidence events."""
    root = _root(mission_id)
    records = read_jsonl(root / "qa.jsonl", QaRecord)
    return {
        "mission_id": mission_id,
        "qa": [row.model_dump(mode="json") for row in records],
        "publishable_qa_ids": [row.qa_id for row in publishable_qa(records)],
    }


@router.get("/{mission_id}/4d/evidence")
def read_evidence(
    mission_id: str,
    event_id: str | None = Query(default=None),
    qa_id: str | None = Query(default=None),
) -> dict:
    """Return the event or QA row and the stored evidence it cites.

    Exactly one of ``event_id`` or ``qa_id`` is required. A missing row is
    HTTP 404. Geometry and mask files are included when they are present.
    """
    if (event_id is None) == (qa_id is None):
        raise HTTPException(status_code=400, detail="provide event_id or qa_id")
    root = _root(mission_id)
    timeline = read_model(root / "timeline.json", SceneTimeline)
    records = read_jsonl(root / "qa.jsonl", QaRecord)
    event = None
    qa = None
    if qa_id is not None:
        qa = next((row for row in records if row.qa_id == qa_id), None)
        if qa is None:
            raise HTTPException(status_code=404, detail="qa not found")
        cited = set(qa.evidence_event_ids)
        events = [row for row in timeline.events if row.event_id in cited]
        if not events:
            raise HTTPException(status_code=404, detail="qa evidence not found")
    else:
        event = next((row for row in timeline.events if row.event_id == event_id), None)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        events = [event]
    evidence = []
    for row in events:
        for item in row.evidence:
            evidence.append(_evidence_payload(root, row.event_id, item))
    return {
        "mission_id": mission_id,
        "event": None if event is None else event.model_dump(mode="json"),
        "qa": None if qa is None else qa.model_dump(mode="json"),
        "publishable": all(row.verification.status == "accepted" for row in events)
        and (qa is None or qa.verification_status == "accepted"),
        "evidence": evidence,
        "publishable_event_ids": [row.event_id for row in publishable_events(events)],
    }


def _evidence_payload(root, event_id: str, item) -> dict:
    payload = {
        "event_id": event_id,
        "frame_id": item.frame_id,
        "t_sec": item.t_sec,
        "track_ids": list(item.track_ids),
        "geometry_ref": item.geometry_ref,
        "mask_ref": item.mask_ref,
        "geometry": None,
        "mask": None,
    }
    if item.geometry_ref is not None:
        path = root / item.geometry_ref
        if path.is_file():
            payload["geometry"] = read_model(path, GeometrySample).model_dump(mode="json")
    if item.mask_ref is not None:
        path = root / item.mask_ref
        if path.is_file():
            payload["mask"] = read_model(path, MaskArtifact).model_dump(mode="json")
    return payload


def _deltas(mission_id: str) -> list[GraphDelta]:
    root = _root(mission_id)
    return read_jsonl(root / "graph-deltas.jsonl", GraphDelta)


def _root(mission_id: str):
    try:
        root = analysis_dir(mission_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="mission not found") from exc
    if not (root / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail="scene graph not found")
    return root
