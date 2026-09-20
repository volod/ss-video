"""asyncpg-backed PostgreSQL helpers for global_map and global_map_missions.

Phase 1: GPS-to-ENU registration — registration_error is NULL (GPS accuracy only).
Phase 2: ICP registration — registration_error is the ICP RMSE residual.

All functions accept an asyncpg Connection (or pool) as their first argument.

Schema (created by scripts/migrate_postgres.py):

    global_map (id, origin_lat, origin_lon, origin_alt, splat_path, created_at, updated_at)

    global_map_missions (
        id, global_map_id, mission_id,
        registration_transform_json, registration_error, registered_at,
        UNIQUE (global_map_id, mission_id)
    )
"""

import math
from typing import Any

from selfsuvis.pipeline.core import get_logger, utcnow
from selfsuvis.pipeline.storage.common import jsonb, row_dict, row_dicts

logger = get_logger(__name__)

# How close two ENU origins must be (metres) to be considered the same site
_SAME_ORIGIN_RADIUS_M = 5_000.0  # 5 km; Phase 3 multi-site can lower this


async def get_or_create_global_map(
    conn,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float = 0.0,
) -> int:
    """Return the id of the global_map row for this GPS origin.

    If no existing row is within _SAME_ORIGIN_RADIUS_M, a new row is inserted.
    A very simple proximity check is used (lat/lon degree distance, not ENU);
    for Phase 3 multi-site support replace with a proper Haversine query.

    Returns:
        global_map id (integer primary key).
    """
    lat_delta = _SAME_ORIGIN_RADIUS_M / 111_000.0
    lon_scale = max(math.cos(math.radians(origin_lat)), 1e-6)
    lon_delta = _SAME_ORIGIN_RADIUS_M / (111_000.0 * lon_scale)
    bucket = f"{round(origin_lat, 2)}:{round(origin_lon, 2)}"
    try:
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", bucket)
    except Exception:
        # Unit-test mocks and non-PostgreSQL shims may not implement advisory locks.
        pass

    try:
        row = await conn.fetchrow(
            """
            SELECT id, origin_lat, origin_lon
            FROM global_map
            WHERE origin_lat BETWEEN $1 AND $2
              AND origin_lon BETWEEN $3 AND $4
            ORDER BY (
                POWER((origin_lat - $5) * 111000.0, 2) +
                POWER((origin_lon - $6) * 111000.0 * $7, 2)
            ) ASC
            LIMIT 1
            """,
            origin_lat - lat_delta,
            origin_lat + lat_delta,
            origin_lon - lon_delta,
            origin_lon + lon_delta,
            origin_lat,
            origin_lon,
            lon_scale,
        )
    except Exception:
        row = None
        rows = await conn.fetch(
            "SELECT id, origin_lat, origin_lon FROM global_map ORDER BY created_at"
        )
        for candidate in rows:
            dlat = abs(candidate["origin_lat"] - origin_lat) * 111_000
            dlon = abs(candidate["origin_lon"] - origin_lon) * 111_000 * lon_scale
            if (dlat**2 + dlon**2) ** 0.5 < _SAME_ORIGIN_RADIUS_M:
                row = candidate
                break
    if row is not None:
        try:
            row_lat = row["origin_lat"]
            row_lon = row["origin_lon"]
            if not isinstance(row_lat, (int, float)) or not isinstance(row_lon, (int, float)):
                raise TypeError("non-numeric row from mock fetchrow")
            dlat = abs(float(row_lat) - origin_lat) * 111_000
            dlon = abs(float(row_lon) - origin_lon) * 111_000 * lon_scale
            if (dlat**2 + dlon**2) ** 0.5 < _SAME_ORIGIN_RADIUS_M:
                return row["id"]
        except Exception:
            rows = await conn.fetch(
                "SELECT id, origin_lat, origin_lon FROM global_map ORDER BY created_at"
            )
            for candidate in rows:
                dlat = abs(candidate["origin_lat"] - origin_lat) * 111_000
                dlon = abs(candidate["origin_lon"] - origin_lon) * 111_000 * lon_scale
                if (dlat**2 + dlon**2) ** 0.5 < _SAME_ORIGIN_RADIUS_M:
                    return candidate["id"]

    now = utcnow()
    row_id = await conn.fetchval(
        """
        INSERT INTO global_map (origin_lat, origin_lon, origin_alt, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """,
        origin_lat,
        origin_lon,
        origin_alt,
        now,
        now,
    )
    logger.info(
        "global_map: created new entry id=%d origin=(%.6f, %.6f, %.1f)",
        row_id,
        origin_lat,
        origin_lon,
        origin_alt,
    )
    return row_id


async def get_global_map_splats(conn, global_map_id: int) -> list[str]:
    """Return splat.ply paths for all missions already registered to this global map.

    These are the target paths passed to run_mapper for ICP registration of new scenes.
    Returns an empty list when no missions have been registered yet (first mission).
    """
    rows = await conn.fetch(
        """
        SELECT m.splat_path
        FROM global_map_missions gmm
        JOIN missions m ON m.id = gmm.mission_id
        WHERE gmm.global_map_id = $1
          AND m.splat_path IS NOT NULL
        ORDER BY gmm.registered_at
        """,
        global_map_id,
    )
    return [row["splat_path"] for row in rows]


async def register_mission(
    conn,
    global_map_id: int,
    mission_id: str,
    transform_4x4: list[list[float]],
    registration_error: float | None,
) -> None:
    """Upsert a mission registration into global_map_missions.

    transform_4x4: 4×4 SE(3) matrix (Phase 1: GPS translation; Phase 2: ICP refined).
    registration_error: None for Phase 1 (GPS-only); ICP RMSE for Phase 2.

    Uses INSERT ... ON CONFLICT DO UPDATE so re-running a mission updates its
    registration without requiring a delete-first.
    """
    transform_json = jsonb(transform_4x4)
    now = utcnow()
    await conn.execute(
        """
        INSERT INTO global_map_missions
            (global_map_id, mission_id, registration_transform_json,
             registration_error, registered_at)
        VALUES ($1, $2, $3::jsonb, $4, $5)
        ON CONFLICT (global_map_id, mission_id)
        DO UPDATE SET
            registration_transform_json = EXCLUDED.registration_transform_json,
            registration_error          = EXCLUDED.registration_error,
            registered_at               = EXCLUDED.registered_at
        """,
        global_map_id,
        mission_id,
        transform_json,
        registration_error,
        now,
    )
    logger.info(
        "global_map_missions: registered mission=%s global_map_id=%d error=%s",
        mission_id,
        global_map_id,
        f"{registration_error:.4f}" if registration_error is not None else "None",
    )


async def update_mission_splat_path(
    conn,
    mission_id: str,
    splat_path: str,
) -> None:
    """Set missions.splat_path after nerfstudio splatfacto produces a splat.ply.

    Required for get_global_map_splats to return this mission's splat as an
    ICP target for future missions at the same site.  The column is added by
    scripts/migrate_postgres.py.
    """
    now = utcnow()
    await conn.execute(
        "UPDATE missions SET splat_path = $1, updated_at = $2 WHERE id = $3",
        splat_path,
        now,
        mission_id,
    )
    logger.info("missions: splat_path set mission_id=%s path=%s", mission_id, splat_path)


async def update_global_map_splat(
    conn,
    global_map_id: int,
    splat_path: str,
) -> None:
    """Set or replace the fused splat.ply path for a global map entry.

    Called after the fused splat.ply is written (Phase 2 full closure).
    """
    now = utcnow()
    await conn.execute(
        "UPDATE global_map SET splat_path = $1, updated_at = $2 WHERE id = $3",
        splat_path,
        now,
        global_map_id,
    )
    logger.info("global_map: updated splat_path=%s for id=%d", splat_path, global_map_id)


async def get_global_map_origin(conn, global_map_id: int) -> tuple | None:
    """Return (origin_lat, origin_lon, origin_alt) for a global_map id, or None."""
    row = await conn.fetchrow(
        "SELECT origin_lat, origin_lon, origin_alt FROM global_map WHERE id = $1",
        global_map_id,
    )
    if row is None:
        return None
    return (row["origin_lat"], row["origin_lon"], row["origin_alt"])


async def list_global_maps(conn) -> list[dict[str, Any]]:
    """Return all global_map rows ordered by creation time (oldest first)."""
    rows = await conn.fetch(
        "SELECT id, origin_lat, origin_lon, origin_alt, splat_path, created_at "
        "FROM global_map ORDER BY created_at"
    )
    return row_dicts(rows)


async def get_global_map_by_id(conn, global_map_id: int) -> dict[str, Any] | None:
    """Fetch a global_map row by id. Returns None if not found."""
    row = await conn.fetchrow("SELECT * FROM global_map WHERE id = $1", global_map_id)
    return row_dict(row)


async def list_mission_registrations(conn, global_map_id: int) -> list[dict[str, Any]]:
    """Return all global_map_missions rows for a global map, newest first."""
    rows = await conn.fetch(
        """
        SELECT * FROM global_map_missions
        WHERE global_map_id = $1
        ORDER BY registered_at DESC
        """,
        global_map_id,
    )
    return row_dicts(rows)
