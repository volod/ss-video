"""Publish video-owned contract messages (camera-event, scene-caption) to MQTT."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from ss_contracts.models import CameraEvent as ContractCameraEvent
from ss_contracts.models import SceneCaption
from ss_kit.logging import get_logger
from ss_kit.mqtt import MqttSettings, TopicBuilder

from .camera_events import CameraEvent
from .camera_settings import camera_settings

logger = get_logger(__name__)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def camera_event_to_contract(event: CameraEvent) -> ContractCameraEvent:
    """Map the local Frigate dataclass onto the ss-common camera-event model."""
    payload = {
        "event_id": event.event_id,
        "camera": event.camera,
        "label": event.label,
        "score": float(event.score),
        "top_score": float(event.top_score),
        "event_type": event.event_type,
        "started_at": event.started_at,
        "ended_at": event.ended_at,
        "has_snapshot": bool(event.has_snapshot),
        "has_clip": bool(event.has_clip),
        "region": dict(event.region or {}),
        "raw": dict(event.raw or {}),
    }
    return ContractCameraEvent.model_validate(payload)


def scene_caption_to_contract(
    *,
    mission_id: str,
    frame_id: str,
    caption: str | None,
    facts_json: dict[str, Any] | None,
    gps_lat: float | None,
    gps_lon: float | None,
    gps_alt: float | None,
    t_sec: float,
    created_at: datetime | None = None,
) -> SceneCaption:
    payload = {
        "mission_id": mission_id,
        "frame_id": frame_id,
        "t_sec": float(t_sec),
        "caption": caption,
        "facts_json": facts_json,
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "gps_alt": gps_alt,
        "created_at": created_at or datetime.now(UTC),
    }
    return SceneCaption.model_validate({k: v for k, v in payload.items() if v is not None})


class VideoContractPublisher:
    """Publish camera-event and scene-caption messages on the committed topic map."""

    def __init__(self, mqtt: MqttSettings | None = None, site_id: str | None = None) -> None:
        self._mqtt = mqtt if mqtt is not None else camera_settings.mqtt
        self._site_id = site_id or camera_settings.site_id
        self._topics = TopicBuilder()
        self._publish: Callable[[str, bytes], Awaitable[None]] | None = None
        self._ready = asyncio.Event()

    def describe(self) -> str:
        return self._mqtt.describe()

    async def run(self, reconnect_interval: float = 5.0) -> None:
        try:
            import aiomqtt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "aiomqtt is required for video contract publishing. "
                "Install it with: pip install aiomqtt"
            ) from exc

        import aiomqtt

        while True:
            try:
                async with aiomqtt.Client(**self._mqtt.client_kwargs()) as client:

                    async def _publish(topic: str, payload: bytes) -> None:
                        await client.publish(topic, payload, qos=1)

                    self._publish = _publish
                    self._ready.set()
                    logger.info("video MQTT publisher connected to %s", self._mqtt.describe())
                    while True:
                        await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._publish = None
                self._ready.clear()
                logger.warning(
                    "video MQTT publisher lost (%s), reconnecting in %ss",
                    exc,
                    reconnect_interval,
                )
                await asyncio.sleep(reconnect_interval)

    async def publish_camera_event(self, event: CameraEvent) -> None:
        model = camera_event_to_contract(event)
        topic = self._topics.build(
            "camera-event", site_id=self._site_id, camera=event.camera or "unknown"
        )
        await self._send(topic, model)

    async def publish_scene_caption(self, caption: SceneCaption) -> None:
        topic = self._topics.build(
            "scene-caption",
            site_id=self._site_id,
            mission_id=caption.mission_id or "unknown",
        )
        await self._send(topic, caption)

    async def publish_caption_row(
        self,
        *,
        mission_id: str,
        frame_id: str,
        caption: str | None,
        facts_json: dict[str, Any] | None,
        gps_lat: float | None,
        gps_lon: float | None,
        gps_alt: float | None,
        t_sec: float,
    ) -> None:
        model = scene_caption_to_contract(
            mission_id=mission_id,
            frame_id=frame_id,
            caption=caption,
            facts_json=facts_json,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            gps_alt=gps_alt,
            t_sec=t_sec,
        )
        await self.publish_scene_caption(model)

    async def _send(self, topic: str, model: Any) -> None:
        publisher = self._publish
        if publisher is None:
            logger.debug("video MQTT publisher not connected; drop %s", topic)
            return
        body = json.dumps(model.model_dump(mode="json", exclude_unset=True)).encode("utf-8")
        try:
            await publisher(topic, body)
        except Exception:
            logger.exception("video MQTT publish failed topic=%s", topic)
