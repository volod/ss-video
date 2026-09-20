"""Consume ss-common contract MQTT messages (and local Frigate events) for the video API."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from ss_contracts.models import CONTRACTS
from ss_kit.logging import get_logger
from ss_kit.mqtt import MqttSettings, TopicBuilder

from .camera_events import CameraEvent, FrigateEventConsumer
from .camera_settings import camera_settings

logger = get_logger(__name__)

OnSensorEvent = Callable[[Any], Awaitable[None]]
OnSensorState = Callable[[Any], Awaitable[None]]
OnCameraEvent = Callable[[CameraEvent], Awaitable[None]]


class ContractEventConsumer:
    """Subscribe to contract sensor topics and optional Frigate camera events.

    Uses ``ss_kit.mqtt.MqttSettings`` (``COOP_MQTT_`` prefix) and the committed topic map.
    """

    def __init__(
        self,
        on_sensor_event: OnSensorEvent | None = None,
        on_sensor_state: OnSensorState | None = None,
        on_camera_event: OnCameraEvent | None = None,
        mqtt: MqttSettings | None = None,
        subscribe_frigate: bool = True,
        subscribe_contracts: bool = True,
    ) -> None:
        self.on_sensor_event = on_sensor_event
        self.on_sensor_state = on_sensor_state
        self.on_camera_event = on_camera_event
        self._mqtt = mqtt if mqtt is not None else camera_settings.mqtt
        self._topics = TopicBuilder()
        self._frigate_prefix = camera_settings.frigate_topic_prefix.rstrip("/")
        self._subscribe_frigate = subscribe_frigate
        self._subscribe_contracts = subscribe_contracts
        self._frigate = FrigateEventConsumer()

    def describe(self) -> str:
        return self._mqtt.describe()

    async def run(self, reconnect_interval: float = 5.0) -> None:
        try:
            import aiomqtt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "aiomqtt is required for the site MQTT consumer. "
                "Install it with: pip install aiomqtt"
            ) from exc

        import aiomqtt

        while True:
            try:
                async with aiomqtt.Client(**self._mqtt.client_kwargs()) as client:
                    logger.info("site MQTT connected to %s", self._mqtt.describe())
                    await self.consume(client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "site MQTT connection lost (%s), reconnecting in %ss", exc, reconnect_interval
                )
                await asyncio.sleep(reconnect_interval)

    async def consume(self, client: Any) -> None:
        if self._subscribe_contracts:
            await client.subscribe(self._topics.subscription("sensor-event"))
            await client.subscribe(self._topics.subscription("sensor-state"))
        if self._subscribe_frigate:
            await client.subscribe(f"{self._frigate_prefix}/events")
            await client.subscribe(f"{self._frigate_prefix}/+/events")
        async for message in client.messages:
            await self.handle_message(str(message.topic), message.payload)

    async def handle_message(self, topic: str, payload: bytes | str) -> None:
        if self._subscribe_contracts:
            resolved = self._topics.resolve(topic)
            if resolved is not None:
                contract_id, _params = resolved
                await self._dispatch_contract(contract_id, payload)
                return
        if self._subscribe_frigate and self._is_frigate_topic(topic):
            event = self._frigate.decode(payload)
            if event and self.on_camera_event:
                try:
                    await self.on_camera_event(event)
                except Exception:
                    logger.exception("Error in on_camera_event callback")

    def _is_frigate_topic(self, topic: str) -> bool:
        return topic.startswith(f"{self._frigate_prefix}/") and topic.endswith("/events")

    async def _dispatch_contract(self, contract_id: str, payload: bytes | str) -> None:
        try:
            raw = json.loads(payload)
            model = CONTRACTS[contract_id].model_validate(raw)
        except Exception:
            logger.exception("Invalid %s payload", contract_id)
            return
        try:
            if contract_id == "sensor-event" and self.on_sensor_event:
                await self.on_sensor_event(model)
            elif contract_id == "sensor-state" and self.on_sensor_state:
                await self.on_sensor_state(model)
        except Exception:
            logger.exception("Error dispatching %s", contract_id)
