"""Queryable 4D metadata stored in the video PostgreSQL database.

Large masks, point clouds, and embeddings stay in the artifact directory.
These tables point at that directory and keep the fields operators filter on.
"""

from typing import Any

from selfsuvis.pipeline.storage.common import jsonb, row_dicts


async def replace_analysis_run(conn, rows: dict[str, Any]) -> None:
    """Insert one analysis run and its event, edge, and QA rows.

    The run id is the manifest digest. Replacing a run that a later run
    supersedes fails at the foreign key, so fast results stay in the table
    when a deep result points at them.

    Args:
        conn: asyncpg connection.
        rows: Output of ``metadata_rows``.
    """
    run = rows["run"]
    await conn.execute("DELETE FROM analysis4d_events WHERE run_id = $1", run["id"])
    await conn.execute("DELETE FROM analysis4d_edges WHERE run_id = $1", run["id"])
    await conn.execute("DELETE FROM analysis4d_qa WHERE run_id = $1", run["id"])
    await conn.execute(
        """
        INSERT INTO analysis4d_runs (
            id, mission_id, profile, schema_version, coordinate_frame, metric_scale,
            calibration_id, artifact_dir, manifest_sha256, degradations_json, supersedes_run_id
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11
        )
        ON CONFLICT (id) DO UPDATE SET
            mission_id = EXCLUDED.mission_id,
            profile = EXCLUDED.profile,
            schema_version = EXCLUDED.schema_version,
            coordinate_frame = EXCLUDED.coordinate_frame,
            metric_scale = EXCLUDED.metric_scale,
            calibration_id = EXCLUDED.calibration_id,
            artifact_dir = EXCLUDED.artifact_dir,
            manifest_sha256 = EXCLUDED.manifest_sha256,
            degradations_json = EXCLUDED.degradations_json,
            supersedes_run_id = EXCLUDED.supersedes_run_id
        """,
        run["id"],
        run["mission_id"],
        run["profile"],
        run["schema_version"],
        run["coordinate_frame"],
        run["metric_scale"],
        run["calibration_id"],
        run["artifact_dir"],
        run["manifest_sha256"],
        jsonb(run["degradations"]),
        run["supersedes_run_id"],
    )
    for event in rows["events"]:
        await conn.execute(
            """
            INSERT INTO analysis4d_events (
                run_id, event_id, mission_id, event_type, summary, start_sec, end_sec,
                verification_status, confidence, participants_json, evidence_count,
                artifact_ref, supersedes_event_id
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12, $13
            )
            """,
            event["run_id"],
            event["event_id"],
            event["mission_id"],
            event["event_type"],
            event["summary"],
            event["start_sec"],
            event["end_sec"],
            event["verification_status"],
            event["confidence"],
            jsonb(event["participants"]),
            event["evidence_count"],
            event["artifact_ref"],
            event["supersedes_event_id"],
        )
    for edge in rows["edges"]:
        await conn.execute(
            """
            INSERT INTO analysis4d_edges (
                run_id, edge_id, mission_id, subject_id, predicate, object_id,
                start_sec, end_sec, coordinate_frame, verification_status,
                confidence, source, artifact_ref
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            )
            """,
            edge["run_id"],
            edge["edge_id"],
            edge["mission_id"],
            edge["subject_id"],
            edge["predicate"],
            edge["object_id"],
            edge["start_sec"],
            edge["end_sec"],
            edge["coordinate_frame"],
            edge["verification_status"],
            edge["confidence"],
            edge["source"],
            edge["artifact_ref"],
        )
    for qa in rows["qa"]:
        await conn.execute(
            """
            INSERT INTO analysis4d_qa (
                run_id, qa_id, mission_id, qa_type, question, answer_kind, answer_value,
                verification_status, interval_start_sec, interval_end_sec,
                evidence_event_ids_json, artifact_ref
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12
            )
            """,
            qa["run_id"],
            qa["qa_id"],
            qa["mission_id"],
            qa["qa_type"],
            qa["question"],
            qa["answer_kind"],
            qa["answer_value"],
            qa["verification_status"],
            qa["interval_start_sec"],
            qa["interval_end_sec"],
            jsonb(qa["evidence_event_ids"]),
            qa["artifact_ref"],
        )


async def list_events(conn, mission_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
    """Return event metadata for a mission, optionally filtered by verification status."""
    if status is None:
        rows = await conn.fetch(
            """
            SELECT * FROM analysis4d_events
            WHERE mission_id = $1
            ORDER BY start_sec, event_id
            """,
            mission_id,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT * FROM analysis4d_events
            WHERE mission_id = $1 AND verification_status = $2
            ORDER BY start_sec, event_id
            """,
            mission_id,
            status,
        )
    return row_dicts(rows)


async def list_edges(
    conn, mission_id: str, *, predicate: str | None = None
) -> list[dict[str, Any]]:
    """Return edge metadata for a mission, optionally filtered by predicate."""
    if predicate is None:
        rows = await conn.fetch(
            """
            SELECT * FROM analysis4d_edges
            WHERE mission_id = $1
            ORDER BY start_sec, edge_id
            """,
            mission_id,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT * FROM analysis4d_edges
            WHERE mission_id = $1 AND predicate = $2
            ORDER BY start_sec, edge_id
            """,
            mission_id,
            predicate,
        )
    return row_dicts(rows)


async def list_qa(conn, mission_id: str) -> list[dict[str, Any]]:
    """Return QA metadata for a mission."""
    rows = await conn.fetch(
        """
        SELECT * FROM analysis4d_qa
        WHERE mission_id = $1
        ORDER BY interval_start_sec, qa_id
        """,
        mission_id,
    )
    return row_dicts(rows)
