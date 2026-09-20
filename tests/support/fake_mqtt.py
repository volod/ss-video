"""Minimal MQTT client stand-in for unit tests (no broker, no aiomqtt)."""

import asyncio
from typing import Any


class FakeMqttMessage:
    def __init__(self, topic: str, payload: bytes) -> None:
        self.topic = topic
        self.payload = payload


class FakeMqttClient:
    """Async context manager with subscribe/publish and an async message stream."""

    def __init__(
        self,
        incoming: list[FakeMqttMessage] | None = None,
        hang: bool = False,
    ) -> None:
        self.subscriptions: list[str] = []
        self.published: list[dict[str, Any]] = []
        self._incoming = list(incoming or [])
        self._hang = hang
        self._closed = asyncio.Event()
        self._iter: list[FakeMqttMessage] = []

    async def __aenter__(self) -> "FakeMqttClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._closed.set()
        return None

    async def subscribe(self, topic: str) -> None:
        self.subscriptions.append(topic)

    async def publish(
        self,
        topic: str,
        payload: bytes = b"",
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        self.published.append({"topic": topic, "payload": payload, "qos": qos, "retain": retain})

    @property
    def messages(self) -> "FakeMqttClient":
        return self

    def __aiter__(self) -> "FakeMqttClient":
        self._iter = list(self._incoming)
        return self

    async def __anext__(self) -> FakeMqttMessage:
        if self._iter:
            return self._iter.pop(0)
        if self._hang:
            await self._closed.wait()
        raise StopAsyncIteration
