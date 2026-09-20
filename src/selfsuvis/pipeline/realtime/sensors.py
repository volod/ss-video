"""Shared sensor registry helpers for realtime ingest."""

from collections.abc import Iterable
from typing import Any

_SUPPORTED_SENSOR_TYPES: set[str] = {
    "barometer",
    "camera",
    "gps",
    "imu",
    "lidar",
    "magnetometer",
}

_SENSOR_CAPABILITIES: dict[str, list[str]] = {
    "barometer": ["altitude"],
    "camera": ["imagery"],
    "gps": ["position", "velocity", "global_reference"],
    "imu": ["orientation", "acceleration", "angular_velocity", "velocity"],
    "lidar": ["depth", "geometry"],
    "magnetometer": ["heading", "orientation_hint"],
}


def normalize_sensor_type(sensor_type: Any) -> str:
    return str(sensor_type or "").strip().lower()


def require_supported_sensor_type(sensor_type: Any) -> str:
    normalized = normalize_sensor_type(sensor_type)
    if normalized not in _SUPPORTED_SENSOR_TYPES:
        raise ValueError(f"unsupported sensor_type: {normalized or '<empty>'}")
    return normalized


def supported_sensor_types() -> set[str]:
    return set(_SUPPORTED_SENSOR_TYPES)


def packet_sensor_summary(sensor_types: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sensor_type in sensor_types:
        key = normalize_sensor_type(sensor_type)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_sensor_profile(sensors: Iterable[Any]) -> dict[str, Any]:
    counts = packet_sensor_summary(sensor for sensor in sensors if normalize_sensor_type(sensor))
    deduped = sorted(counts)
    return {
        "sensors": deduped,
        "sensor_count": len(deduped),
        "capabilities": {sensor: list(_SENSOR_CAPABILITIES.get(sensor, [])) for sensor in deduped},
    }
