"""Shared helpers for telemetry-to-packet media bridges."""

from collections.abc import Awaitable, Callable, Iterable
from typing import Any


def build_packet(
    *,
    sensor_type: str,
    t_device: float,
    payload: dict[str, Any],
    seq: int | None = None,
) -> dict[str, Any]:
    return {
        "sensor_type": str(sensor_type).strip().lower(),
        "t_device": float(t_device),
        "seq": int(seq) if seq is not None else None,
        "payload": dict(payload or {}),
    }


def flatten_packet_batches(
    messages: Iterable[dict[str, Any]],
    converter: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for message in messages:
        packets.extend(converter(message))
    return packets


class PacketBridge:
    """Shared async bridge that converts external messages into realtime packets."""

    def __init__(
        self,
        converter: Callable[[dict[str, Any]], list[dict[str, Any]]],
        *,
        on_packet: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._converter = converter
        self._on_packet = on_packet

    async def ingest_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        packets = self._converter(message)
        if self._on_packet is not None:
            for packet in packets:
                await self._on_packet(packet)
        return packets

    async def ingest_messages(self, messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for message in messages:
            collected.extend(await self.ingest_message(message))
        return collected
