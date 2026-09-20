"""PostgreSQL helpers for mission/frame metadata persistence."""

from collections.abc import Iterable
from typing import Any

from selfsuvis.pipeline.core import to_utc_datetime, utcnow
from selfsuvis.pipeline.storage.common import (
    jsonb,
    jsonb_optional,
    qdrant_id_to_pg,
    row_dict,
    row_dicts,
)


async def upsert_mission(
    conn,
    *,
    mission_id: str,
    video_id: str,
    video_path: str,
    job_id: str,
    robot_id: str,
    status: str,
    frame_count: int,
    duration_sec: float | None,
    gps_origin: dict[str, Any] | None,
    pose_status: str = "pending",
    map_status: str = "pending",
    error: str | None = None,
) -> None:
    now = utcnow()
    await conn.execute(
        """
        INSERT INTO missions
            (id, video_id, video_path, job_id, robot_id, status, pose_status, map_status,
             frame_count, duration_sec, gps_origin_json, created_at, updated_at, error)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8,
             $9, $10, $11::jsonb, $12, $13, $14)
        ON CONFLICT (id) DO UPDATE SET
            video_id = EXCLUDED.video_id,
            video_path = EXCLUDED.video_path,
            job_id = EXCLUDED.job_id,
            robot_id = EXCLUDED.robot_id,
            status = EXCLUDED.status,
            pose_status = EXCLUDED.pose_status,
            map_status = EXCLUDED.map_status,
            frame_count = EXCLUDED.frame_count,
            duration_sec = EXCLUDED.duration_sec,
            gps_origin_json = EXCLUDED.gps_origin_json,
            updated_at = EXCLUDED.updated_at,
            error = EXCLUDED.error
        """,
        mission_id,
        video_id,
        video_path,
        job_id,
        robot_id,
        status,
        pose_status,
        map_status,
        frame_count,
        duration_sec,
        jsonb_optional(gps_origin),
        now,
        now,
        error,
    )


async def fetch_mission(conn, mission_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT id, video_id, video_path, job_id, robot_id, status, pose_status, map_status,
               frame_count, duration_sec, gps_origin_json, created_at, updated_at, error
        FROM missions
        WHERE id = $1
        """,
        mission_id,
    )
    return row_dict(row)


async def replace_frames(conn, mission_id: str, frames: Iterable[dict[str, Any]]) -> None:
    rows = list(frames)
    await conn.execute("DELETE FROM frames WHERE mission_id = $1", mission_id)
    if not rows:
        return

    now = utcnow()
    await conn.executemany(
        """
        INSERT INTO frames
            (id, mission_id, frame_path, t_sec, segment_id, caption, caption_confidence,
             caption_model, subtitle_text, ocr_text, frame_facts_json,
             al_score, al_tag, cvat_label, pose_status, pose_json, gps_json,
             global_pose_json, qdrant_id, created_at, updated_at)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7,
             $8, $9, $10, $11::jsonb,
             $12, $13, $14, $15, $16::jsonb, $17::jsonb,
             $18::jsonb, $19, $20, $21)
        """,
        [
            (
                row["id"],
                mission_id,
                row["frame_path"],
                row["t_sec"],
                row.get("segment_id"),
                row.get("caption"),
                row.get("caption_confidence"),
                row.get("caption_model"),
                row.get("subtitle_text"),
                row.get("ocr_text"),
                jsonb_optional(row.get("frame_facts_json")),
                row.get("al_score"),
                row.get("al_tag", "none"),
                row.get("cvat_label"),
                row.get("pose_status", "pending"),
                jsonb_optional(row.get("pose_json")),
                jsonb_optional(row.get("gps_json")),
                jsonb_optional(row.get("global_pose_json")),
                qdrant_id_to_pg(row.get("qdrant_id")),
                now,
                now,
            )
            for row in rows
        ],
    )


async def list_mission_frames(conn, mission_id: str) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id, mission_id, frame_path, t_sec, segment_id, caption, caption_confidence,
               caption_model, subtitle_text, ocr_text, frame_facts_json, al_score, al_tag,
               cvat_label, pose_status, pose_json, gps_json, global_pose_json, qdrant_id,
               created_at, updated_at
        FROM frames
        WHERE mission_id = $1
        ORDER BY t_sec ASC, id ASC
        """,
        mission_id,
    )
    return row_dicts(rows)


async def mark_mission_finished(
    conn,
    mission_id: str,
    *,
    status: str,
    pose_status: str | None = None,
    map_status: str | None = None,
    error: str | None = None,
) -> None:
    assignments = ["status = $1", "updated_at = $2", "error = $3"]
    values: list[Any] = [status, utcnow(), error]
    idx = 4
    if pose_status is not None:
        assignments.append(f"pose_status = ${idx}")
        values.append(pose_status)
        idx += 1
    if map_status is not None:
        assignments.append(f"map_status = ${idx}")
        values.append(map_status)
        idx += 1
    values.append(mission_id)
    await conn.execute(
        f"UPDATE missions SET {', '.join(assignments)} WHERE id = ${idx}",
        *values,
    )


async def apply_gps_registration(
    conn,
    mission_id: str,
    enu_origin: dict[str, Any] | None,
    global_poses: dict[str, Any],
) -> None:
    now = utcnow()
    if enu_origin is not None:
        await conn.execute(
            "UPDATE missions SET gps_origin_json = $1::jsonb, updated_at = $2 WHERE id = $3",
            jsonb(enu_origin),
            now,
            mission_id,
        )
    if not global_poses or not isinstance(global_poses, dict):
        return
    await conn.executemany(
        """
        UPDATE frames
        SET global_pose_json = $1::jsonb,
            updated_at = $2
        WHERE id = $3
        """,
        [(jsonb(pose), now, frame_id) for frame_id, pose in global_poses.items()],
    )


async def list_frames_after(conn, cursor: tuple | None, limit: int) -> list[dict[str, Any]]:
    if cursor is None:
        rows = await conn.fetch(
            """
            SELECT id, qdrant_id, frame_path, mission_id, created_at
            FROM frames
            ORDER BY created_at ASC, id ASC
            LIMIT $1
            """,
            limit,
        )
    else:
        created_at, frame_id = cursor
        rows = await conn.fetch(
            """
            SELECT id, qdrant_id, frame_path, mission_id, created_at
            FROM frames
            WHERE (created_at, id) > ($1, $2)
            ORDER BY created_at ASC, id ASC
            LIMIT $3
            """,
            to_utc_datetime(created_at),
            frame_id,
            limit,
        )
    return row_dicts(rows)
