"""PostgreSQL helpers for realtime drone sessions, poses, tiles, and semantics."""

from collections.abc import Iterable
from typing import Any

from selfsuvis.pipeline.core import utcnow
from selfsuvis.pipeline.storage.common import jsonb, jsonb_optional, row_dict, row_dicts


async def create_robot_session(
    conn,
    *,
    session_id: str,
    robot_id: str,
    mission_id: str | None = None,
    sensor_profile: dict[str, Any] | None = None,
    status: str = "active",
) -> None:
    now = utcnow()
    await conn.execute(
        """
        INSERT INTO robot_sessions
            (id, robot_id, mission_id, sensor_profile_json, status, started_at, updated_at)
        VALUES
            ($1, $2, $3, $4::jsonb, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE SET
            robot_id = EXCLUDED.robot_id,
            mission_id = EXCLUDED.mission_id,
            sensor_profile_json = EXCLUDED.sensor_profile_json,
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at
        """,
        session_id,
        robot_id,
        mission_id,
        jsonb(sensor_profile, default={}),
        status,
        now,
        now,
    )


async def stop_robot_session(conn, session_id: str, status: str = "stopped") -> None:
    now = utcnow()
    await conn.execute(
        """
        UPDATE robot_sessions
        SET status = $1, ended_at = $2, updated_at = $2
        WHERE id = $3
        """,
        status,
        now,
        session_id,
    )


async def fetch_robot_session(conn, session_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT id, robot_id, mission_id, sensor_profile_json, status,
               started_at, ended_at, updated_at
        FROM robot_sessions
        WHERE id = $1
        """,
        session_id,
    )
    return row_dict(row)


async def insert_sensor_packets(
    conn,
    session_id: str,
    packets: Iterable[dict[str, Any]],
) -> int:
    rows = list(packets)
    if not rows:
        return 0
    now = utcnow()
    await conn.executemany(
        """
        INSERT INTO sensor_packets
            (session_id, sensor_type, t_device, t_server, seq, payload_json)
        VALUES
            ($1, $2, $3, $4, $5, $6::jsonb)
        """,
        [
            (
                session_id,
                row["sensor_type"],
                row["t_device"],
                now,
                row.get("seq"),
                jsonb(row.get("payload"), default={}),
            )
            for row in rows
        ],
    )
    return len(rows)


async def insert_realtime_pose(
    conn,
    *,
    session_id: str,
    source: str,
    t_sec: float,
    position_enu: dict[str, float],
    orientation_quat: dict[str, float] | None = None,
    velocity_enu: dict[str, float] | None = None,
    covariance: dict[str, Any] | None = None,
    tracking_status: str = "ok",
    global_map_id: int | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO realtime_poses
            (session_id, source, t_sec, position_enu_json, orientation_quat_json,
             velocity_enu_json, covariance_json, tracking_status, global_map_id)
        VALUES
            ($1, $2, $3, $4::jsonb, $5::jsonb,
             $6::jsonb, $7::jsonb, $8, $9)
        """,
        session_id,
        source,
        t_sec,
        jsonb(position_enu),
        jsonb_optional(orientation_quat),
        jsonb_optional(velocity_enu),
        jsonb_optional(covariance),
        tracking_status,
        global_map_id,
    )


async def fetch_latest_realtime_pose(conn, session_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT id, session_id, source, t_sec, position_enu_json, orientation_quat_json,
               velocity_enu_json, covariance_json, tracking_status, global_map_id, created_at
        FROM realtime_poses
        WHERE session_id = $1
        ORDER BY t_sec DESC, id DESC
        LIMIT 1
        """,
        session_id,
    )
    return row_dict(row)


async def fetch_realtime_state(conn, session_id: str) -> dict[str, Any] | None:
    session = await fetch_robot_session(conn, session_id)
    if session is None:
        return None
    latest_pose = await fetch_latest_realtime_pose(conn, session_id)
    packet_counts = await conn.fetch(
        """
        SELECT sensor_type, COUNT(*)::BIGINT AS n
        FROM sensor_packets
        WHERE session_id = $1
        GROUP BY sensor_type
        ORDER BY sensor_type
        """,
        session_id,
    )
    return {
        "session": session,
        "latest_pose": latest_pose,
        "packet_counts": {row["sensor_type"]: int(row["n"]) for row in packet_counts},
    }


async def summarize_realtime_session(conn, session_id: str) -> dict[str, Any] | None:
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        return None
    row = await conn.fetchrow(
        """
        SELECT MIN(t_device) AS t_min, MAX(t_device) AS t_max
        FROM sensor_packets
        WHERE session_id = $1
        """,
        session_id,
    )
    t_min = row["t_min"] if row else None
    t_max = row["t_max"] if row else None
    duration_sec = None
    if t_min is not None and t_max is not None:
        duration_sec = max(0.0, float(t_max) - float(t_min))
    return {
        **state,
        "duration_sec": duration_sec,
    }


async def upsert_map_tile(
    conn,
    *,
    session_id: str,
    tile_key: str,
    map_type: str,
    storage_path: str,
    resolution_m: float,
    bounds: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    global_map_id: int | None = None,
) -> None:
    now = utcnow()
    await conn.execute(
        """
        INSERT INTO map_tiles
            (session_id, global_map_id, tile_key, map_type, storage_path,
             resolution_m, bounds_json, stats_json, updated_at)
        VALUES
            ($1, $2, $3, $4, $5,
             $6, $7::jsonb, $8::jsonb, $9)
        ON CONFLICT (session_id, tile_key, map_type) DO UPDATE SET
            global_map_id = EXCLUDED.global_map_id,
            storage_path = EXCLUDED.storage_path,
            resolution_m = EXCLUDED.resolution_m,
            bounds_json = EXCLUDED.bounds_json,
            stats_json = EXCLUDED.stats_json,
            updated_at = EXCLUDED.updated_at
        """,
        session_id,
        global_map_id,
        tile_key,
        map_type,
        storage_path,
        resolution_m,
        jsonb(bounds, default={}),
        jsonb(stats, default={}),
        now,
    )


async def list_map_tiles(
    conn,
    session_id: str,
    map_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if map_type:
        rows = await conn.fetch(
            """
            SELECT id, session_id, global_map_id, tile_key, map_type, storage_path,
                   resolution_m, bounds_json, stats_json, updated_at
            FROM map_tiles
            WHERE session_id = $1 AND map_type = $2
            ORDER BY updated_at DESC, id DESC
            LIMIT $3
            """,
            session_id,
            map_type,
            limit,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT id, session_id, global_map_id, tile_key, map_type, storage_path,
                   resolution_m, bounds_json, stats_json, updated_at
            FROM map_tiles
            WHERE session_id = $1
            ORDER BY updated_at DESC, id DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
    return row_dicts(rows)


async def insert_semantic_observation(
    conn,
    *,
    session_id: str,
    class_name: str,
    confidence: float,
    position_enu: dict[str, Any] | None = None,
    bbox: dict[str, Any] | None = None,
    frame_id: str | None = None,
    mask_ref: str | None = None,
    track_id: str | None = None,
    facts: dict[str, Any] | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO semantic_observations
            (session_id, frame_id, class_name, confidence, position_enu_json,
             bbox_json, mask_ref, track_id, facts_json)
        VALUES
            ($1, $2, $3, $4, $5::jsonb,
             $6::jsonb, $7, $8, $9::jsonb)
        """,
        session_id,
        frame_id,
        class_name,
        confidence,
        jsonb_optional(position_enu),
        jsonb_optional(bbox),
        mask_ref,
        track_id,
        jsonb(facts, default={}),
    )


async def upsert_realtime_frame(
    conn,
    *,
    session_id: str,
    frame_id: str,
    t_sec: float,
    image_path: str,
    pose: dict[str, Any] | None = None,
    depth_path: str | None = None,
    tile_key: str | None = None,
    map_type: str = "occupancy",
    stats: dict[str, Any] | None = None,
) -> None:
    now = utcnow()
    await conn.execute(
        """
        INSERT INTO realtime_frames
            (session_id, frame_id, t_sec, image_path, pose_json, depth_path, tile_key, map_type, stats_json, updated_at)
        VALUES
            ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9::jsonb, $10)
        ON CONFLICT (session_id, frame_id) DO UPDATE SET
            t_sec = EXCLUDED.t_sec,
            image_path = EXCLUDED.image_path,
            pose_json = EXCLUDED.pose_json,
            depth_path = EXCLUDED.depth_path,
            tile_key = EXCLUDED.tile_key,
            map_type = EXCLUDED.map_type,
            stats_json = EXCLUDED.stats_json,
            updated_at = EXCLUDED.updated_at
        """,
        session_id,
        frame_id,
        t_sec,
        image_path,
        jsonb_optional(pose),
        depth_path,
        tile_key,
        map_type,
        jsonb(stats, default={}),
        now,
    )


async def list_realtime_frames(conn, session_id: str, limit: int = 10000) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id, session_id, frame_id, t_sec, image_path, pose_json, depth_path,
               tile_key, map_type, stats_json, updated_at
        FROM realtime_frames
        WHERE session_id = $1
        ORDER BY t_sec ASC, id ASC
        LIMIT $2
        """,
        session_id,
        limit,
    )
    return row_dicts(rows)


async def list_semantic_observations(
    conn,
    session_id: str,
    class_name: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if class_name:
        rows = await conn.fetch(
            """
            SELECT id, session_id, frame_id, class_name, confidence, position_enu_json,
                   bbox_json, mask_ref, track_id, facts_json, created_at
            FROM semantic_observations
            WHERE session_id = $1 AND class_name = $2
            ORDER BY created_at DESC, id DESC
            LIMIT $3
            """,
            session_id,
            class_name,
            limit,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT id, session_id, frame_id, class_name, confidence, position_enu_json,
                   bbox_json, mask_ref, track_id, facts_json, created_at
            FROM semantic_observations
            WHERE session_id = $1
            ORDER BY created_at DESC, id DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
    return row_dicts(rows)
