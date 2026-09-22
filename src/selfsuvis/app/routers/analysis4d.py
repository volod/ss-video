"""Read APIs for a mission's temporal scene graph.

These routes expose graph deltas, the materialized view, and VLM proposals.
Timeline and Video-QA routes are a later task.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.pipeline.analysis4d.graph import materialize
from selfsuvis.pipeline.analysis4d.io import read_jsonl
from selfsuvis.pipeline.analysis4d.paths import analysis_dir
from selfsuvis.pipeline.analysis4d.schemas import GraphDelta, Proposal

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
