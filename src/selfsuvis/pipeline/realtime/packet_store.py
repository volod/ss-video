"""Persist normalized realtime packets and optional fused pose."""

from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.realtime.ingest import normalize_packets
from selfsuvis.pipeline.realtime.pose import RealtimePoseClient, build_fused_pose_from_packets
from selfsuvis.pipeline.realtime.sensors import packet_sensor_summary
from selfsuvis.pipeline.storage.realtime import (
    fetch_realtime_state,
    insert_realtime_pose,
    insert_sensor_packets,
)


async def _require_realtime_state(conn, session_id: str) -> dict[str, Any]:
    state = await fetch_realtime_state(conn, session_id)
    if state is None:
        raise LookupError("session not found")
    return state


def _pose_client() -> RealtimePoseClient | None:
    if settings.REALTIME_POSE_BACKEND == "stub":
        return None
    return RealtimePoseClient()


async def estimate_realtime_pose(
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
        return {
            "source": latest_pose_row["source"],
            "t_sec": float(latest_pose_row["t_sec"]),
            "position_enu": dict(latest_pose_row["position_enu_json"]),
            "orientation_quat": dict(latest_pose_row["orientation_quat_json"])
            if latest_pose_row.get("orientation_quat_json")
            else None,
            "velocity_enu": dict(latest_pose_row["velocity_enu_json"])
            if latest_pose_row.get("velocity_enu_json")
            else None,
            "covariance": dict(latest_pose_row["covariance_json"])
            if latest_pose_row.get("covariance_json")
            else None,
            "tracking_status": latest_pose_row["tracking_status"],
            "global_map_id": latest_pose_row.get("global_map_id"),
        }
    return None


async def ingest_realtime_packets(
    conn, *, session_id: str, packets: list[dict[str, Any]]
) -> dict[str, Any]:
    normalized = normalize_packets(packets)
    if len(normalized) > settings.REALTIME_PACKET_BATCH_SIZE:
        raise ValueError(f"too many packets: max {settings.REALTIME_PACKET_BATCH_SIZE}")
    await _require_realtime_state(conn, session_id)
    await insert_sensor_packets(conn, session_id, normalized)

    packet_summary = packet_sensor_summary(packet["sensor_type"] for packet in normalized)
    pose = await estimate_realtime_pose(session_id=session_id, packets=normalized)
    pose_updated = False
    if pose is not None:
        await insert_realtime_pose(
            conn,
            session_id=session_id,
            source=pose["source"],
            t_sec=pose["t_sec"],
            position_enu=pose["position_enu"],
            orientation_quat=pose["orientation_quat"],
            velocity_enu=pose["velocity_enu"],
            covariance=pose["covariance"],
            tracking_status=pose["tracking_status"],
            global_map_id=pose["global_map_id"],
        )
        pose_updated = True
    return {
        "session_id": session_id,
        "accepted_packets": len(normalized),
        "packet_summary": packet_summary,
        "pose_updated": pose_updated,
    }
