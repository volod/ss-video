"""Drone telemetry bridge helpers."""

from collections.abc import Iterable
from typing import Any

from .bridge_common import PacketBridge, build_packet, flatten_packet_batches
from .mavlink import mavlink_message_to_packets


def bridge_mavlink_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return flatten_packet_batches(messages, mavlink_message_to_packets)


def frame_capture_to_packet(
    *,
    t_device: float,
    frame_id: str,
    image_path: str,
    width: int,
    height: int,
    camera_id: str = "front",
) -> dict[str, Any]:
    return build_packet(
        sensor_type="camera",
        t_device=t_device,
        payload={
            "frame_id": str(frame_id),
            "image_path": str(image_path),
            "width": int(width),
            "height": int(height),
            "camera_id": str(camera_id),
        },
    )


class DroneTelemetryBridge(PacketBridge):
    """Bridge decoded drone telemetry into repo-native realtime packets."""

    def __init__(self, on_packet=None) -> None:
        super().__init__(mavlink_message_to_packets, on_packet=on_packet)
