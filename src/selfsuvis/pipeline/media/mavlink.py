"""Lightweight MAVLink packet normalization helpers.

These helpers intentionally operate on decoded dict-like payloads so the bridge
can be tested without a hard MAVSDK / pymavlink dependency.
"""

from typing import Any

from .bridge_common import build_packet


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mavlink_message_to_packets(message: dict[str, Any]) -> list[dict[str, Any]]:
    kind = str(message.get("message_type") or message.get("type") or "").strip().upper()
    t_device = _f(message.get("t_device") or message.get("timestamp") or message.get("time_usec"))
    if t_device is None:
        raise ValueError("mavlink message missing timestamp")
    if t_device > 1_000_000_000_000:
        t_device /= 1_000_000.0
    elif t_device > 1_000_000_000:
        t_device /= 1000.0

    if kind in {"GLOBAL_POSITION_INT", "GPS_RAW_INT"}:
        lat = _f(message.get("lat"))
        lon = _f(message.get("lon"))
        alt = _f(
            message.get("relative_alt")
            if message.get("relative_alt") is not None
            else message.get("alt")
        )
        east = _f(message.get("east"))
        north = _f(message.get("north"))
        packet = build_packet(
            sensor_type="gps",
            t_device=t_device,
            seq=message.get("seq"),
            payload={
                "lat": lat / 1e7 if lat is not None and abs(lat) > 180 else lat,
                "lon": lon / 1e7 if lon is not None and abs(lon) > 180 else lon,
                "altitude": (alt / 1000.0) if alt is not None and abs(alt) > 1000 else alt,
                "east": east,
                "north": north,
                "up": _f(message.get("up")),
                "vx": (_f(message.get("vx")) or 0.0) / 100.0
                if message.get("vx") is not None
                else None,
                "vy": (_f(message.get("vy")) or 0.0) / 100.0
                if message.get("vy") is not None
                else None,
                "vz": (_f(message.get("vz")) or 0.0) / 100.0
                if message.get("vz") is not None
                else None,
                "global_map_id": message.get("global_map_id"),
            },
        )
        return [packet]

    if kind in {"ATTITUDE", "HIGHRES_IMU", "RAW_IMU"}:
        packet = build_packet(
            sensor_type="imu",
            t_device=t_device,
            seq=message.get("seq"),
            payload={
                "roll": _f(message.get("roll")),
                "pitch": _f(message.get("pitch")),
                "yaw": _f(message.get("yaw")),
                "angular_velocity": {
                    "x": _f(message.get("rollspeed")),
                    "y": _f(message.get("pitchspeed")),
                    "z": _f(message.get("yawspeed")),
                },
                "acceleration": {
                    "x": _f(message.get("xacc")),
                    "y": _f(message.get("yacc")),
                    "z": _f(message.get("zacc")),
                },
                "global_map_id": message.get("global_map_id"),
            },
        )
        return [packet]

    if kind in {"SCALED_PRESSURE", "ALTITUDE"}:
        return [
            build_packet(
                sensor_type="barometer",
                t_device=t_device,
                seq=message.get("seq"),
                payload={
                    "altitude": _f(
                        message.get("altitude")
                        if message.get("altitude") is not None
                        else message.get("press_abs")
                    ),
                    "global_map_id": message.get("global_map_id"),
                },
            )
        ]

    if kind in {"VFR_HUD", "COMPASSMOT_STATUS"} and message.get("heading") is not None:
        return [
            build_packet(
                sensor_type="magnetometer",
                t_device=t_device,
                seq=message.get("seq"),
                payload={
                    "heading": _f(message.get("heading")),
                    "global_map_id": message.get("global_map_id"),
                },
            )
        ]

    return []
