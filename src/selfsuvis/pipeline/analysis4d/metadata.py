"""PostgreSQL rows for queryable 4D event, edge, and QA metadata."""

from typing import Any

from selfsuvis.pipeline.analysis4d.io import sha256_bytes
from selfsuvis.pipeline.analysis4d.validate import Bundle


def run_id_for(manifest_bytes: bytes) -> str:
    """Return the run id, which is the manifest digest without the ``sha256:`` prefix."""
    return sha256_bytes(manifest_bytes).removeprefix("sha256:")


def metadata_rows(bundle: Bundle, *, artifact_dir: str, manifest_bytes: bytes) -> dict[str, Any]:
    """Build the query rows for one validated bundle.

    Args:
        bundle: Validated mission.
        artifact_dir: Relative or absolute artifact directory stored on the run row.
        manifest_bytes: Canonical manifest bytes. Their digest is the run id.

    Returns:
        A dict with ``run``, ``events``, ``edges``, and ``qa`` row lists.
    """
    manifest = bundle.manifest
    run_id = run_id_for(manifest_bytes)
    run = {
        "id": run_id,
        "mission_id": manifest.mission_id,
        "profile": manifest.profile,
        "schema_version": manifest.schema_version,
        "coordinate_frame": manifest.coordinate_frame.name,
        "metric_scale": manifest.coordinate_frame.metric_scale,
        "calibration_id": manifest.coordinate_frame.calibration_id,
        "artifact_dir": artifact_dir,
        "manifest_sha256": "sha256:" + run_id,
        "degradations": [item.model_dump(mode="json") for item in manifest.degradations],
        "supersedes_run_id": (
            None
            if manifest.supersedes_sha256 is None
            else manifest.supersedes_sha256.removeprefix("sha256:")
        ),
    }
    events = [
        {
            "run_id": run_id,
            "event_id": event.event_id,
            "mission_id": manifest.mission_id,
            "event_type": event.type,
            "summary": event.summary,
            "start_sec": event.start_sec,
            "end_sec": event.end_sec,
            "verification_status": event.verification.status,
            "confidence": event.confidence,
            "participants": event.participants,
            "evidence_count": len(event.evidence),
            "artifact_ref": "timeline.json",
            "supersedes_event_id": event.supersedes,
        }
        for event in bundle.timeline.events
    ]
    edges = []
    for delta in bundle.deltas:
        if delta.edge is None:
            continue
        edge = delta.edge
        edges.append(
            {
                "run_id": run_id,
                "edge_id": edge.edge_id,
                "mission_id": manifest.mission_id,
                "subject_id": edge.subject_id,
                "predicate": edge.predicate,
                "object_id": edge.object_id,
                "start_sec": edge.start_sec,
                "end_sec": edge.end_sec,
                "coordinate_frame": edge.coordinate_frame,
                "verification_status": edge.verification_status,
                "confidence": edge.confidence,
                "source": edge.source,
                "artifact_ref": "graph-deltas.jsonl",
            }
        )
    qa_rows = [
        {
            "run_id": run_id,
            "qa_id": pair.qa_id,
            "mission_id": manifest.mission_id,
            "qa_type": pair.type,
            "question": pair.question,
            "answer_kind": pair.answer.kind,
            "answer_value": None if pair.answer.value is None else str(pair.answer.value),
            "verification_status": pair.verification_status,
            "interval_start_sec": pair.interval_sec[0],
            "interval_end_sec": pair.interval_sec[1],
            "evidence_event_ids": pair.evidence_event_ids,
            "artifact_ref": "qa.jsonl",
        }
        for pair in bundle.qa
    ]
    return {"run": run, "events": events, "edges": edges, "qa": qa_rows}
