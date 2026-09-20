"""ROS-style message normalization for realtime ingest."""

from typing import Any

from .bridge_common import PacketBridge, build_packet


def ros_message_to_packets(message: dict[str, Any]) -> list[dict[str, Any]]:
    topic = str(message.get("topic") or "").strip()
    t_device = float(message.get("t_device") or message.get("stamp") or 0.0)
    payload = dict(message.get("payload") or {})

    if topic.endswith("/imu"):
        return [build_packet(sensor_type="imu", t_device=t_device, payload=payload)]
    if topic.endswith("/gps") or topic.endswith("/fix"):
        return [build_packet(sensor_type="gps", t_device=t_device, payload=payload)]
    if topic.endswith("/barometer") or topic.endswith("/pressure"):
        return [build_packet(sensor_type="barometer", t_device=t_device, payload=payload)]
    if topic.endswith("/mag") or topic.endswith("/magnetometer"):
        return [build_packet(sensor_type="magnetometer", t_device=t_device, payload=payload)]
    if topic.endswith("/image") or topic.endswith("/camera/image_raw"):
        return [build_packet(sensor_type="camera", t_device=t_device, payload=payload)]
    return []


class RosTopicBridge(PacketBridge):
    """Bridge ROS-style topic messages into repo-native realtime packets."""

    def __init__(self, on_packet=None) -> None:
        super().__init__(ros_message_to_packets, on_packet=on_packet)
