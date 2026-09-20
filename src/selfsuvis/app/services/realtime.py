"""Service layer for realtime session, pose, tile, and semantic workflows."""

import uuid
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.core.log_analytics import get_log_analytics
from selfsuvis.pipeline.realtime import (
    build_fused_pose_from_packets,
    build_sensor_profile,
    new_session_id,
    normalize_map_tile,
    normalize_packets,
    normalize_pose_payload,
    normalize_semantic_observation,
    project_detection_to_enu,
)
from selfsuvis.pipeline.realtime.occupancy import RealtimeOccupancyClient, write_stub_map_tile
from selfsuvis.pipeline.realtime.packet_store import (
    ingest_realtime_packets as _ingest_realtime_packets,
)
from selfsuvis.pipeline.realtime.pose import RealtimePoseClient
from selfsuvis.pipeline.storage.jobs import create_job
from selfsuvis.pipeline.storage.missions import upsert_mission
from selfsuvis.pipeline.storage.realtime import (
    create_robot_session,
    fetch_realtime_state,
    insert_realtime_pose,
    insert_semantic_observation,
    insert_sensor_packets,
    list_map_tiles,
    list_realtime_frames,
    list_semantic_observations,
    stop_robot_session,
    summarize_realtime_session,
    upsert_map_tile,
    upsert_realtime_frame,
)
from selfsuvis.pipeline.vision.depth import DepthModel
from selfsuvis.realtime.adapters import describe_occupancy_backends, describe_pose_backends

ingest_realtime_packets = _ingest_realtime_packets


async def _require_realtime_state(conn, session_id: str) -> dict[str, Any]:
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise LookupError("session not found")
    return state


async def _store_session_payload(
    conn,
    *,
    session_id: str,
    payload: dict[str, Any],
    normalize_fn,
    store_fn,
) -> None:
    normalized = normalize_fn(payload)
    await _require_realtime_state(conn, session_id)
    await store_fn(conn, session_id=session_id, **normalized)


async def _list_session_payloads(
    conn, *, session_id: str, list_fn, **kwargs: Any
) -> list[dict[str, Any]]:
    await _require_realtime_state(conn, session_id)
    return await list_fn(conn, session_id, **kwargs)


def _pose_client() -> RealtimePoseClient | None:
    if settings.REALTIME_POSE_BACKEND == "stub":
        return None
    return RealtimePoseClient()


def _occupancy_client() -> RealtimeOccupancyClient | None:
    if settings.REALTIME_OCCUPANCY_BACKEND == "stub":
        return None
    return RealtimeOccupancyClient()


def _row_to_pose_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": row["source"],
        "t_sec": float(row["t_sec"]),
        "position_enu": dict(row["position_enu_json"]),
        "orientation_quat": dict(row["orientation_quat_json"])
        if row.get("orientation_quat_json")
        else None,
        "velocity_enu": dict(row["velocity_enu_json"]) if row.get("velocity_enu_json") else None,
        "covariance": dict(row["covariance_json"]) if row.get("covariance_json") else None,
        "tracking_status": row["tracking_status"],
        "global_map_id": row.get("global_map_id"),
    }


async def _estimate_realtime_pose(
    *,
    session_id: str,
    packets: list[dict[str, Any]],
    latest_pose_row: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    pose_client = _pose_client()
    if pose_client is not None and pose_client.is_configured and packets:
        try:
            pose = await pose_client.estimate_pose(session_id=session_id, packets=packets)
        except Exception:
            pose = None
        if pose is not None:
            return pose
    if packets:
        pose = build_fused_pose_from_packets(
            packets,
            max_lag_ms=settings.REALTIME_MAX_SENSOR_LAG_MS,
        )
        if pose is not None:
            return pose
    if latest_pose_row is not None:
        return _row_to_pose_payload(latest_pose_row)
    return None


async def _collect_client_stats(client) -> dict[str, Any]:
    if client is None:
        return {}
    try:
        return await client.stats()
    except Exception as exc:
        return {"configured": client.is_configured, "error": str(exc)}


async def start_realtime_session(
    conn,
    *,
    robot_id: str,
    mission_id: str | None,
    sensors: list[str],
) -> dict[str, Any]:
    session_id = new_session_id()
    sensor_profile = build_sensor_profile(sensors)
    await create_robot_session(
        conn,
        session_id=session_id,
        robot_id=robot_id,
        mission_id=mission_id,
        sensor_profile=sensor_profile,
    )
    return {
        "session_id": session_id,
        "robot_id": robot_id,
        "mission_id": mission_id,
        "sensor_profile": sensor_profile,
        "status": "active",
    }


async def collect_realtime_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {
        "pose_backend": settings.REALTIME_POSE_BACKEND,
        "occupancy_backend": settings.REALTIME_OCCUPANCY_BACKEND,
        "logging": get_log_analytics().snapshot(),
    }
    pose_stats = await _collect_client_stats(_pose_client())
    occ_stats = await _collect_client_stats(_occupancy_client())
    if pose_stats:
        stats["pose"] = pose_stats
    if occ_stats:
        stats["occupancy"] = occ_stats
    return stats


def list_realtime_backends() -> dict[str, Any]:
    return {
        "selected": {
            "pose_backend": settings.REALTIME_POSE_BACKEND,
            "occupancy_backend": settings.REALTIME_OCCUPANCY_BACKEND,
        },
        "pose_backends": describe_pose_backends(),
        "occupancy_backends": describe_occupancy_backends(),
    }


async def integrate_realtime_frame(
    conn,
    *,
    session_id: str,
    frame_id: str | None,
    t_sec: float,
    image_path: str,
    packets: list[dict[str, Any]] | None = None,
    semantic_observations: list[dict[str, Any]] | None = None,
    pose: dict[str, Any] | None = None,
    depth_path: str | None = None,
    map_type: str = "occupancy",
    tile_key: str | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = await _require_realtime_state(conn, session_id)

    normalized_packets: list[dict[str, Any]] = []
    if packets:
        normalized_packets = normalize_packets(packets)
        await insert_sensor_packets(conn, session_id, normalized_packets)

    pose_payload = normalize_pose_payload(pose) if pose is not None else None
    if pose_payload is None:
        pose_payload = await _estimate_realtime_pose(
            session_id=session_id,
            packets=normalized_packets,
            latest_pose_row=state.get("latest_pose"),
        )

    pose_updated = False
    if pose_payload is not None:
        await insert_realtime_pose(
            conn,
            session_id=session_id,
            source=pose_payload["source"],
            t_sec=pose_payload["t_sec"],
            position_enu=pose_payload["position_enu"],
            orientation_quat=pose_payload.get("orientation_quat"),
            velocity_enu=pose_payload.get("velocity_enu"),
            covariance=pose_payload.get("covariance"),
            tracking_status=pose_payload.get("tracking_status", "ok"),
            global_map_id=pose_payload.get("global_map_id"),
        )
        pose_updated = True

    dense_depth_path = depth_path
    if dense_depth_path is None and settings.DEPTH_OUTPUT_MODE == "dense":
        try:
            import numpy as np
            from PIL import Image

            from selfsuvis.pipeline.realtime.occupancy import realtime_tile_dir

            image = Image.open(image_path).convert("RGB")
            depth_model = DepthModel()
            dense = depth_model.estimate_dense(image)
            depth_model.release()
            depth_payload = dense.get("depth_dense") if isinstance(dense, dict) else None
            if isinstance(depth_payload, dict) and depth_payload.get("map") is not None:
                depth_dir = realtime_tile_dir(session_id, map_type="depth")
                dense_depth_path = str(depth_dir / f"frame_{int(round(t_sec * 1000.0))}.npz")
                np.savez_compressed(
                    dense_depth_path,
                    depth=np.asarray(depth_payload["map"], dtype=np.float32),
                    confidence=np.asarray(depth_payload["confidence"], dtype=np.float32),
                )
        except Exception:
            dense_depth_path = None

    occ_payload = {
        "session_id": session_id,
        "frame_id": frame_id,
        "t_sec": float(t_sec),
        "image_path": image_path,
        "depth_path": dense_depth_path,
        "map_type": map_type,
        "tile_key": tile_key,
        "resolution_m": settings.REALTIME_OCCUPANCY_RESOLUTION_M,
        "pose": pose_payload,
        "stats": stats or {},
    }
    occ_client = _occupancy_client()
    tile = None
    if occ_client is not None and occ_client.is_configured:
        try:
            tile = await occ_client.integrate_frame(occ_payload)
        except Exception:
            tile = None
    if tile is None:
        tile = write_stub_map_tile(
            session_id=session_id,
            t_sec=t_sec,
            frame_id=frame_id,
            map_type=map_type,
            resolution_m=settings.REALTIME_OCCUPANCY_RESOLUTION_M,
            pose=pose_payload,
            stats=stats,
        )
    await upsert_map_tile(conn, session_id=session_id, **tile)

    published_semantics = 0
    for observation in semantic_observations or []:
        if (
            pose_payload is not None
            and not observation.get("position_enu")
            and observation.get("bbox")
        ):
            projected = project_detection_to_enu(
                pose=pose_payload,
                bbox=observation["bbox"],
                range_m=(observation.get("facts") or {}).get("range_m"),
            )
            if projected is not None:
                observation = {**observation, "position_enu": projected}
        normalized_obs = normalize_semantic_observation(observation)
        await insert_semantic_observation(conn, session_id=session_id, **normalized_obs)
        published_semantics += 1

    if frame_id:
        await upsert_realtime_frame(
            conn,
            session_id=session_id,
            frame_id=frame_id,
            t_sec=t_sec,
            image_path=image_path,
            pose=pose_payload,
            depth_path=dense_depth_path,
            tile_key=tile["tile_key"],
            map_type=tile["map_type"],
            stats=stats,
        )

    return {
        "session_id": session_id,
        "pose_updated": pose_updated,
        "tile": tile,
        "semantic_count": published_semantics,
        "depth_path": dense_depth_path,
    }


async def publish_map_tile(conn, *, session_id: str, tile: dict[str, Any]) -> None:
    await _store_session_payload(
        conn,
        session_id=session_id,
        payload=tile,
        normalize_fn=normalize_map_tile,
        store_fn=upsert_map_tile,
    )


async def fetch_map_tiles(
    conn,
    *,
    session_id: str,
    map_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return await _list_session_payloads(
        conn,
        session_id=session_id,
        list_fn=list_map_tiles,
        map_type=map_type,
        limit=limit,
    )


async def publish_semantic_observation(
    conn, *, session_id: str, observation: dict[str, Any]
) -> None:
    await _store_session_payload(
        conn,
        session_id=session_id,
        payload=observation,
        normalize_fn=normalize_semantic_observation,
        store_fn=insert_semantic_observation,
    )


async def fetch_semantic_observations(
    conn,
    *,
    session_id: str,
    class_name: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return await _list_session_payloads(
        conn,
        session_id=session_id,
        list_fn=list_semantic_observations,
        class_name=class_name,
        limit=limit,
    )


async def finalize_realtime_session(
    conn,
    *,
    session_id: str,
    recording_path: str | None = None,
    enqueue_index_job: bool = False,
) -> dict[str, Any]:
    state = await _require_realtime_state(conn, session_id)
    session = state["session"]
    mission_id = session.get("mission_id") or f"realtime-{session_id}"
    video_id = mission_id
    summary = await summarize_realtime_session(conn, session_id)
    realtime_frames = await list_realtime_frames(conn, session_id)
    frame_count = int(summary.get("packet_counts", {}).get("camera", 0))
    if realtime_frames:
        frame_count = max(frame_count, len(realtime_frames))
    duration_sec = summary.get("duration_sec")
    latest_pose = summary.get("latest_pose")
    gps_origin = None
    if latest_pose and latest_pose.get("position_enu_json"):
        gps_origin = {"enu_origin_hint": dict(latest_pose["position_enu_json"])}

    await upsert_mission(
        conn,
        mission_id=mission_id,
        video_id=video_id,
        video_path=recording_path or "",
        job_id="",
        robot_id=session["robot_id"],
        status="pending",
        frame_count=frame_count,
        duration_sec=duration_sec,
        gps_origin=gps_origin,
    )

    job_id: str | None = None
    if enqueue_index_job and recording_path:
        job_id = uuid.uuid4().hex
        await create_job(
            conn,
            job_id,
            {
                "video_id": video_id,
                "video_path": recording_path,
                "mission_id": mission_id,
                "enable_tiles": True,
                "realtime_session_id": session_id,
                "postflight_jobs": [
                    "postflight_mapping",
                    "postflight_semantic_graph",
                ],
            },
            job_type="index",
        )
        await upsert_mission(
            conn,
            mission_id=mission_id,
            video_id=video_id,
            video_path=recording_path,
            job_id=job_id,
            robot_id=session["robot_id"],
            status="pending",
            frame_count=frame_count,
            duration_sec=duration_sec,
            gps_origin=gps_origin,
        )

    if realtime_frames and not recording_path:
        from selfsuvis.pipeline.storage.missions import replace_frames

        rows = []
        for idx, frame in enumerate(realtime_frames):
            frame_row = {
                "id": f"{mission_id}:{idx}:{int(float(frame['t_sec']) * 1000)}",
                "frame_path": frame["image_path"],
                "t_sec": float(frame["t_sec"]),
                "segment_id": idx,
                "caption": None,
                "caption_confidence": None,
                "caption_model": None,
                "subtitle_text": None,
                "ocr_text": None,
                "frame_facts_json": {
                    "realtime_frame": True,
                    "tile_key": frame.get("tile_key"),
                    "map_type": frame.get("map_type"),
                    "stats": dict(frame.get("stats_json") or {}),
                },
                "al_score": None,
                "al_tag": "none",
                "pose_status": "success" if frame.get("pose_json") else "pending",
                "pose_json": dict(frame["pose_json"]) if frame.get("pose_json") else None,
                "gps_json": None,
                "global_pose_json": None,
                "qdrant_id": None,
            }
            rows.append(frame_row)
        await replace_frames(conn, mission_id, rows)

    await stop_robot_session(conn, session_id)
    return {
        "session_id": session_id,
        "mission_id": mission_id,
        "job_id": job_id,
        "status": "stopped",
        "enqueued_index_job": bool(job_id),
    }
