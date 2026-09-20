"""Realtime pose helpers.

The current implementation performs a lightweight batch fusion pass over recent
sensor packets. It is not a full EKF, but it produces a better pose estimate
than the older GPS-only fallback by combining GPS position, IMU orientation and
velocity, barometric altitude, and magnetometer heading when those signals are
fresh enough to trust together.
"""

import math
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from selfsuvis.pipeline.core import settings
from selfsuvis.pipeline.realtime.sidecar import RealtimeSidecarClient
from selfsuvis.realtime.adapters import create_pose_adapter


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _quat_from_yaw(yaw_rad: float) -> dict[str, float]:
    half = yaw_rad / 2.0
    return {
        "x": 0.0,
        "y": 0.0,
        "z": math.sin(half),
        "w": math.cos(half),
    }


def _quat_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float) -> dict[str, float]:
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    return {
        "w": cr * cp * cy + sr * sp * sy,
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
    }


def _payload_global_map_id(payload: dict[str, Any]) -> int | None:
    value = payload.get("global_map_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gps_pose(packet: dict[str, Any]) -> dict[str, Any] | None:
    payload = packet.get("payload") or {}
    east = _coerce_float(payload.get("east"))
    north = _coerce_float(payload.get("north"))
    if east is None or north is None:
        return None
    up = _coerce_float(payload.get("up"))
    if up is None:
        up = _coerce_float(payload.get("altitude"))
    velocity = None
    for keys in (("vx", "vy", "vz"), ("ve", "vn", "vu")):
        vx = _coerce_float(payload.get(keys[0]))
        vy = _coerce_float(payload.get(keys[1]))
        vz = _coerce_float(payload.get(keys[2]))
        if vx is not None and vy is not None:
            velocity = {"x": vx, "y": vy, "z": vz or 0.0}
            break
    return {
        "t_sec": float(packet["t_device"]),
        "position_enu": {"x": east, "y": north, "z": up or 0.0},
        "velocity_enu": velocity,
        "global_map_id": _payload_global_map_id(payload),
    }


def _imu_orientation(payload: dict[str, Any]) -> dict[str, float] | None:
    quat = payload.get("orientation_quat")
    if isinstance(quat, dict):
        x = _coerce_float(quat.get("x"))
        y = _coerce_float(quat.get("y"))
        z = _coerce_float(quat.get("z"))
        w = _coerce_float(quat.get("w"))
        if None not in (x, y, z, w):
            return {"x": x, "y": y, "z": z, "w": w}
    roll = _coerce_float(payload.get("roll"))
    pitch = _coerce_float(payload.get("pitch"))
    yaw = _coerce_float(payload.get("yaw"))
    if yaw is not None:
        return _quat_from_euler(roll or 0.0, pitch or 0.0, yaw)
    return None


def _imu_pose(packet: dict[str, Any]) -> dict[str, Any] | None:
    payload = packet.get("payload") or {}
    orientation = _imu_orientation(payload)
    velocity = payload.get("velocity_enu")
    velocity_enu = None
    if isinstance(velocity, dict):
        vx = _coerce_float(velocity.get("x"))
        vy = _coerce_float(velocity.get("y"))
        vz = _coerce_float(velocity.get("z"))
        if vx is not None and vy is not None:
            velocity_enu = {"x": vx, "y": vy, "z": vz or 0.0}
    if velocity_enu is None:
        vx = _coerce_float(payload.get("vx"))
        vy = _coerce_float(payload.get("vy"))
        vz = _coerce_float(payload.get("vz"))
        if vx is not None and vy is not None:
            velocity_enu = {"x": vx, "y": vy, "z": vz or 0.0}
    if orientation is None and velocity_enu is None:
        return None
    return {
        "t_sec": float(packet["t_device"]),
        "orientation_quat": orientation,
        "velocity_enu": velocity_enu,
        "global_map_id": _payload_global_map_id(payload),
    }


def _barometer_pose(packet: dict[str, Any]) -> dict[str, Any] | None:
    payload = packet.get("payload") or {}
    altitude = _coerce_float(payload.get("altitude"))
    if altitude is None:
        altitude = _coerce_float(payload.get("up"))
    if altitude is None:
        return None
    return {
        "t_sec": float(packet["t_device"]),
        "altitude": altitude,
        "global_map_id": _payload_global_map_id(payload),
    }


def _magnetometer_pose(packet: dict[str, Any]) -> dict[str, Any] | None:
    payload = packet.get("payload") or {}
    heading = _coerce_float(payload.get("heading"))
    if heading is None:
        heading = _coerce_float(payload.get("yaw"))
    if heading is None:
        return None
    return {
        "t_sec": float(packet["t_device"]),
        "orientation_quat": _quat_from_yaw(heading),
        "global_map_id": _payload_global_map_id(payload),
    }


def _trace_covariance(modalities: list[str], time_offsets_ms: dict[str, int]) -> dict[str, Any]:
    freshness_penalty = sum(max(offset, 0) for offset in time_offsets_ms.values()) / 1000.0
    base = max(0.05, 1.0 - min(len(modalities), 4) * 0.18)
    return {
        "trace": round(base + freshness_penalty, 4),
        "modalities": list(modalities),
        "time_offsets_ms": dict(time_offsets_ms),
    }


def build_fused_pose_from_packets(
    packets: Iterable[dict[str, Any]],
    *,
    max_lag_ms: int,
) -> dict[str, Any] | None:
    sorted_packets = sorted(packets, key=lambda packet: float(packet["t_device"]))
    if not sorted_packets:
        return None

    latest_by_type: dict[str, dict[str, Any]] = {}
    for packet in sorted_packets:
        latest_by_type[packet["sensor_type"]] = packet

    gps = _gps_pose(latest_by_type["gps"]) if "gps" in latest_by_type else None
    imu = _imu_pose(latest_by_type["imu"]) if "imu" in latest_by_type else None
    barometer = (
        _barometer_pose(latest_by_type["barometer"]) if "barometer" in latest_by_type else None
    )
    magnetometer = (
        _magnetometer_pose(latest_by_type["magnetometer"])
        if "magnetometer" in latest_by_type
        else None
    )
    if gps is None and imu is None and barometer is None and magnetometer is None:
        return None

    anchors = [item["t_sec"] for item in (gps, imu, barometer, magnetometer) if item is not None]
    anchor_t = max(anchors)

    def is_fresh(item: dict[str, Any] | None) -> bool:
        if item is None:
            return False
        return abs(anchor_t - float(item["t_sec"])) * 1000.0 <= max_lag_ms

    gps_fresh = is_fresh(gps)
    imu_fresh = is_fresh(imu)
    barometer_fresh = is_fresh(barometer)
    magnetometer_fresh = is_fresh(magnetometer)

    if not gps_fresh and gps is not None:
        gps = None
    if not imu_fresh and imu is not None:
        imu = None
    if not barometer_fresh and barometer is not None:
        barometer = None
    if not magnetometer_fresh and magnetometer is not None:
        magnetometer = None

    if gps is None:
        return None

    position_enu = dict(gps["position_enu"])
    if barometer is not None:
        position_enu["z"] = float(barometer["altitude"])

    orientation = None
    if imu is not None:
        orientation = imu.get("orientation_quat")
    if orientation is None and magnetometer is not None:
        orientation = magnetometer.get("orientation_quat")

    velocity = None
    if imu is not None:
        velocity = imu.get("velocity_enu")
    if velocity is None:
        velocity = gps.get("velocity_enu")

    modalities = ["gps"]
    if imu is not None:
        modalities.append("imu")
    if barometer is not None:
        modalities.append("barometer")
    if magnetometer is not None and orientation is not None and imu is None:
        modalities.append("magnetometer")

    time_offsets_ms = {
        sensor: int(round(abs(anchor_t - float(item["t_sec"])) * 1000.0))
        for sensor, item in (
            ("gps", gps),
            ("imu", imu),
            ("barometer", barometer),
            ("magnetometer", magnetometer),
        )
        if item is not None
    }

    source = "gps_fallback" if modalities == ["gps"] else f"fused_{'_'.join(modalities)}"
    tracking_status = "degraded" if modalities == ["gps"] else "ok"
    global_map_id = next(
        (
            item.get("global_map_id")
            for item in (gps, imu, barometer, magnetometer)
            if item is not None and item.get("global_map_id") is not None
        ),
        None,
    )
    return {
        "source": source,
        "t_sec": anchor_t,
        "position_enu": position_enu,
        "orientation_quat": orientation,
        "velocity_enu": velocity,
        "covariance": _trace_covariance(modalities, time_offsets_ms),
        "tracking_status": tracking_status,
        "global_map_id": global_map_id,
    }


def build_stub_pose_from_packet(packet: dict[str, Any]) -> dict[str, Any] | None:
    return build_fused_pose_from_packets([packet], max_lag_ms=0)


def normalize_pose_payload(payload: dict[str, Any]) -> dict[str, Any]:
    position = dict(payload.get("position_enu") or {})
    orientation = payload.get("orientation_quat")
    velocity = payload.get("velocity_enu")
    covariance = payload.get("covariance")
    if "x" not in position or "y" not in position:
        raise ValueError("pose payload requires position_enu.x and position_enu.y")
    return {
        "source": str(payload.get("source") or "sidecar"),
        "t_sec": float(payload["t_sec"]),
        "position_enu": {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position.get("z", 0.0) or 0.0),
        },
        "orientation_quat": dict(orientation) if isinstance(orientation, dict) else None,
        "velocity_enu": dict(velocity) if isinstance(velocity, dict) else None,
        "covariance": dict(covariance) if isinstance(covariance, dict) else None,
        "tracking_status": str(payload.get("tracking_status") or "ok"),
        "global_map_id": _payload_global_map_id(payload),
    }


class RealtimePoseClient(RealtimeSidecarClient):
    """HTTP client for external realtime pose backends."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_sec: float = 5.0,
    ) -> None:
        adapter = create_pose_adapter(settings.REALTIME_POSE_BACKEND)
        resolved_url = base_url or settings.REALTIME_POSE_API_URL or adapter.api_url
        super().__init__(
            backend_name=adapter.name,
            base_url=resolved_url,
            timeout_sec=timeout_sec,
        )

    async def estimate_pose(
        self,
        *,
        session_id: str,
        packets: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.is_configured:
            return None
        payload = {"session_id": session_id, "packets": packets}
        data = await self.request_json("POST", "/estimate_pose", payload=payload)
        pose = self.unwrap_dict_payload(data, field="pose")
        if not isinstance(pose, dict):
            return None
        return normalize_pose_payload(pose)


def pose_freshness_ms(created_at: Any, now: datetime | None = None) -> int | None:
    if created_at is None:
        return None
    if isinstance(created_at, str):
        created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    else:
        created_dt = created_at
    if created_dt.tzinfo is None:
        created_dt = created_dt.replace(tzinfo=timezone.utc)
    now_dt = now or datetime.now(timezone.utc)
    return max(0, int((now_dt - created_dt).total_seconds() * 1000))
