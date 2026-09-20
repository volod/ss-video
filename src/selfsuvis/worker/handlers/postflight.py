"""Postflight job handlers: mapping (Pass A) and semantic graph.

Also contains the shared postflight-chain utilities (_enqueue_postflight_jobs,
_finalize_postflight_job_success/error) and the site-ENU helpers
(_resolve_site_origin, _run_pass_a) that the INDEX handler also imports.
"""

import os
import time
import uuid

import asyncpg

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.storage import create_job, update_job
from selfsuvis.pipeline.storage.missions import (
    apply_gps_registration,
    fetch_mission,
    list_mission_frames,
    mark_mission_finished,
)
from selfsuvis.worker._run import _run, _update_job_sync

# -- Postflight job chaining --------------------------------------------------


def _normalize_postflight_job_names(value) -> list[str]:
    names = []
    for item in value or []:
        name = str(item or "").strip()
        if not name:
            continue
        if name not in names:
            names.append(name)
    return names


_VALID_POSTFLIGHT_TYPES = {"postflight_mapping", "postflight_semantic_graph"}


async def _enqueue_postflight_jobs(conn, payload: dict, logger) -> list[str]:
    requested = _normalize_postflight_job_names(payload.get("postflight_jobs"))
    if not requested:
        return []

    created: list[str] = []
    base_payload = {
        "video_id": payload.get("video_id"),
        "video_path": payload.get("video_path"),
        "mission_id": payload.get("mission_id") or payload.get("video_id"),
        "realtime_session_id": payload.get("realtime_session_id"),
    }
    remaining = list(requested)
    while remaining:
        job_name = remaining.pop(0)
        next_jobs = list(remaining)
        if job_name not in _VALID_POSTFLIGHT_TYPES:
            logger.warning("Skipping unknown postflight job type=%s", job_name)
            continue
        child_job_id = uuid.uuid4().hex
        child_payload = {**base_payload, "next_postflight_jobs": next_jobs}
        await create_job(conn, child_job_id, child_payload, job_type=job_name)
        created.append(child_job_id)
        logger.info(
            "Post-flight job enqueued parent_mission=%s job_id=%s type=%s",
            base_payload["mission_id"],
            child_job_id,
            job_name,
        )
        break
    return created


async def _finalize_postflight_job_success(
    conn,
    *,
    job_id: str,
    mission_id: str,
    payload: dict,
    progress: dict,
    logger,
) -> None:
    await update_job(
        conn,
        job_id,
        status="finished",
        progress=progress,
        finished_at=time.time(),
    )
    next_payload = {
        **payload,
        "postflight_jobs": payload.get("next_postflight_jobs", []),
    }
    created = await _enqueue_postflight_jobs(conn, next_payload, logger)
    if not created:
        await mark_mission_finished(conn, mission_id, status="done", error=None)


async def _finalize_postflight_job_error(
    conn,
    *,
    job_id: str,
    mission_id: str,
    error: str,
) -> None:
    await update_job(
        conn,
        job_id,
        status="error",
        error=error,
        finished_at=time.time(),
    )
    await mark_mission_finished(conn, mission_id, status="error", error=error)


# -- Site ENU origin resolution (shared with INDEX handler) -------------------


def _resolve_site_origin(video_path: str, logger) -> tuple:
    """Extract the first GPS fix from a video and look up (or create) its global_map.

    Returns (global_map_id, (origin_lat, origin_lon, origin_alt)) on success,
    or (None, None) if GPS is unavailable or DB is unreachable.

    Runs synchronously before index_video so that ENU coordinates stored
    in Qdrant are relative to the site's canonical ENU origin rather than
    each mission's own local first-frame origin.
    """
    try:
        from selfsuvis.pipeline.media.gps import extract_gps

        gps_list = extract_gps(video_path, [1_000.0])
        first_gps = next((g for g in gps_list if g is not None), None)
        if first_gps is None:
            logger.debug("Multi-site ENU: no GPS in video path=%s", video_path)
            return None, None
    except Exception as exc:
        logger.debug("Multi-site ENU: GPS extraction failed path=%s err=%s", video_path, exc)
        return None, None

    try:
        from selfsuvis.pipeline.storage.global_maps import (
            get_global_map_origin,
            get_or_create_global_map,
        )

        async def _lookup():
            conn = await asyncpg.connect(settings.DATABASE_URL)
            try:
                gmap_id = await get_or_create_global_map(
                    conn,
                    first_gps["lat"],
                    first_gps["lon"],
                    float(first_gps.get("alt", 0.0)),
                )
                origin = await get_global_map_origin(conn, gmap_id)
                return gmap_id, origin
            finally:
                await conn.close()

        gmap_id, origin = _run(_lookup())
        if origin is not None:
            logger.info(
                "Multi-site ENU: video assigned to global_map_id=%d origin=(%.6f, %.6f, %.1f)",
                gmap_id,
                origin[0],
                origin[1],
                origin[2],
            )
        return gmap_id, origin
    except Exception as exc:
        logger.debug("Multi-site ENU: site origin lookup failed: %s", exc)
        return None, None


# -- Pass A: SfM -> GPS registration -> 3DGS (shared with INDEX handler) -----


def _run_pass_a(
    video_path: str,
    video_id: str,
    mission_id: str,
    index_result: dict,
    logger,
    global_map_id: int = None,
) -> None:
    """Run SfM -> GPS registration -> nerfstudio 3DGS for a completed indexing job.

    All steps degrade gracefully: missing optional deps (pycolmap, open3d) or
    unreachable containers (nerfstudio, mapper, postgres) are logged and skipped.
    """
    try:
        from selfsuvis.pipeline.mapping.sfm import run_sfm
    except ImportError:
        logger.debug("Pass A: pycolmap not installed -- skipping SfM")
        return

    try:
        sfm_out = run_sfm(video_path, video_id, mission_id)
    except Exception as exc:
        logger.warning("Pass A: SfM failed mission=%s: %s", mission_id, exc)
        return

    sfm_results = sfm_out.get("frames", [])
    scene_count = sfm_out.get("scene_count", 1)
    logger.info(
        "Pass A: SfM done mission=%s scene_count=%d registered=%d",
        mission_id,
        scene_count,
        sum(1 for r in sfm_results if r.get("pose_status") == "success"),
    )

    try:
        from selfsuvis.pipeline.mapping.gps_registration import register_mission_gps

        keyed_frames = sfm_results
        if index_result.get("frame_records"):
            keyed_frames = []
            indexed_by_t = {
                round(frame["t_sec"], 3): frame for frame in index_result.get("frame_records", [])
            }
            for row in sfm_results:
                keyed = dict(row)
                matched = indexed_by_t.get(round(row.get("t_sec", 0.0), 3))
                keyed["gps_json"] = matched.get("gps_json") if matched else None
                keyed["id"] = matched.get("id") if matched else None
                keyed_frames.append(keyed)
        enu_origin, global_poses = register_mission_gps(keyed_frames)
        logger.info(
            "Pass A: GPS registration done mission=%s enu_origin=%s", mission_id, enu_origin
        )
    except Exception as exc:
        logger.warning("Pass A: GPS registration failed mission=%s: %s", mission_id, exc)
        enu_origin = None
        global_poses = {}

    try:
        from selfsuvis.pipeline.mapping.mapper import run_mapper
        from selfsuvis.pipeline.storage.global_maps import (
            get_global_map_splats,
            get_or_create_global_map,
            register_mission,
            update_global_map_splat,
            update_mission_splat_path,
        )
    except ImportError as exc:
        logger.debug("Pass A: mapper deps unavailable (%s) -- skipping 3DGS", exc)
        return

    async def _db_and_map():
        nonlocal global_map_id
        conn = await asyncpg.connect(settings.DATABASE_URL)
        try:
            await mark_mission_finished(
                conn,
                mission_id,
                status="indexing",
                pose_status="running",
                map_status="running",
            )
            if index_result.get("frame_records"):
                await apply_gps_registration(conn, mission_id, enu_origin, global_poses)

            target_splat_paths = []
            if global_map_id is None and enu_origin is not None:
                if isinstance(enu_origin, dict):
                    lat, lon, alt = enu_origin["lat"], enu_origin["lon"], enu_origin.get("alt", 0.0)
                else:
                    lat, lon, alt = enu_origin
                global_map_id = await get_or_create_global_map(conn, lat, lon, alt)
            if global_map_id is not None:
                target_splat_paths = await get_global_map_splats(conn, global_map_id)

            mapper_result = run_mapper(
                mission_id,
                sfm_results,
                scene_count=scene_count,
                target_splat_paths=target_splat_paths,
            )
            logger.info(
                "Pass A: mapper done mission=%s status=%s splats=%d icp=%d",
                mission_id,
                mapper_result["map_status"],
                len(mapper_result.get("splat_paths", [])),
                len(mapper_result.get("icp_results", [])),
            )

            primary_splat = mapper_result.get("splat_path")

            if primary_splat is not None and global_map_id is not None:
                await update_mission_splat_path(conn, mission_id, primary_splat)

            if global_map_id is not None:
                icp_registered = False
                for icp in mapper_result.get("icp_results", []):
                    if icp.get("converged"):
                        await register_mission(
                            conn,
                            global_map_id,
                            mission_id,
                            icp.get("transform_4x4"),
                            icp.get("rmse"),
                        )
                        icp_registered = True
                        if icp.get("fused_splat"):
                            await update_global_map_splat(conn, global_map_id, icp["fused_splat"])

                if not icp_registered and primary_splat is not None:
                    _identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                    await register_mission(conn, global_map_id, mission_id, _identity, None)

            await mark_mission_finished(
                conn,
                mission_id,
                status="indexing",
                pose_status="success" if sfm_results else "skipped",
                map_status=mapper_result["map_status"],
            )
        finally:
            await conn.close()

    try:
        _run(_db_and_map())
    except Exception as exc:
        logger.warning("Pass A: mapper/DB step failed mission=%s: %s", mission_id, exc)


# -- Postflight job handlers --------------------------------------------------


def handle_postflight_mapping_job(job_id: str, payload: dict, pool, logger) -> None:
    video_path = payload.get("video_path")
    video_id = payload.get("video_id")
    mission_id = payload.get("mission_id") or video_id
    if not video_path or not video_id or not mission_id:
        _update_job_sync(
            pool,
            job_id,
            status="error",
            error="postflight_mapping requires video_id, mission_id, and video_path",
            finished_at=time.time(),
        )
        return

    try:
        if not os.path.exists(video_path):
            raise RuntimeError("video_path not found")

        async def _load_frames():
            async with pool.acquire() as conn:
                return await list_mission_frames(conn, mission_id)

        frame_records = _run(_load_frames())
        global_map_id, _site_enu_origin = _resolve_site_origin(video_path, logger)
        _run_pass_a(
            video_path,
            video_id,
            mission_id,
            {"frame_records": frame_records},
            logger,
            global_map_id=global_map_id,
        )

        async def _finish():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_success(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        payload=payload,
                        progress={"mission_id": mission_id, "video_id": video_id},
                        logger=logger,
                    )

        _run(_finish())
        logger.info("Post-flight mapping job finished id=%s mission=%s", job_id, mission_id)
    except Exception as exc:
        logger.exception("Post-flight mapping job failed id=%s error=%s", job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_error(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        error=error_message,
                    )

        _run(_mark_error())


def handle_postflight_semantic_graph_job(job_id: str, payload: dict, pool, logger) -> None:
    mission_id = payload.get("mission_id") or payload.get("video_id")
    if not mission_id:
        _update_job_sync(
            pool,
            job_id,
            status="error",
            error="postflight_semantic_graph requires mission_id",
            finished_at=time.time(),
        )
        return

    try:
        from selfsuvis.pipeline.mapping import (
            build_semantic_environment_graph,
            write_semantic_graph_markdown,
        )

        async def _load_mission_state():
            async with pool.acquire() as conn:
                mission = await fetch_mission(conn, mission_id)
                frames = await list_mission_frames(conn, mission_id)
                return mission, frames

        mission, frames = _run(_load_mission_state())
        if mission is None:
            raise LookupError(f"mission not found: {mission_id}")

        graph_dir = os.path.join(settings.MAPS_DIR, mission_id)
        graph_json = os.path.join(graph_dir, "semantic_environment_graph.json")
        graph_md = os.path.join(graph_dir, "semantic_environment_graph.md")
        graph = build_semantic_environment_graph(
            frames,
            graph_id=mission_id,
            output_path=graph_json,
        )
        write_semantic_graph_markdown(
            graph,
            graph_md,
            title=f"{mission_id} -- Semantic Environment Graph",
        )

        async def _finish():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_success(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        payload=payload,
                        progress={
                            "mission_id": mission_id,
                            "graph_json": graph_json,
                            "graph_markdown": graph_md,
                            "node_count": graph.get("summary", {}).get("node_count", 0),
                            "edge_count": graph.get("summary", {}).get("edge_count", 0),
                        },
                        logger=logger,
                    )

        _run(_finish())
        logger.info("Post-flight semantic graph job finished id=%s mission=%s", job_id, mission_id)
    except Exception as exc:
        logger.exception("Post-flight semantic graph job failed id=%s error=%s", job_id, exc)
        error_message = str(exc)

        async def _mark_error():
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _finalize_postflight_job_error(
                        conn,
                        job_id=job_id,
                        mission_id=mission_id,
                        error=error_message,
                    )

        _run(_mark_error())
