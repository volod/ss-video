"""Publish video-owned contract messages to MQTT.

The long-lived client publishes ``camera-event`` and ``scene-caption``. Accepted
4D timelines publish ``event-envelope`` messages whose payload is
``verified-scene-event`` 1.0.0. The topic modality is ``video_4d``.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from selfsuvis.pipeline.analysis4d.verified_scene_event import VerifiedSceneEvent
from ss_contracts.models import CameraEvent as ContractCameraEvent
from ss_contracts.models import EventEnvelope, SceneCaption
from ss_kit.logging import get_logger
from ss_kit.mqtt import MqttSettings, TopicBuilder

from .camera_events import CameraEvent
from .camera_settings import camera_settings

logger = get_logger(__name__)

VERIFIED_EVENT_MODALITY = "video_4d"
_EVENT_ENVELOPE = "event-envelope"
_ONCE_TIMEOUT_SEC = 2.0


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
    """Publish camera-event, scene-caption, and verified 4D envelopes."""

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

    async def publish_verified_event(self, envelope: EventEnvelope) -> bool:
        """Publish one accepted verified-scene-event envelope.

        Args:
            envelope: Site ``event-envelope`` whose payload is the 4D contract.

        Returns:
            True when the client seam was asked to send the message. False when
            the payload is not an accepted verified scene event.
        """
        payload = envelope.payload or {}
        if payload.get("verification_status") != "accepted":
            return False
        VerifiedSceneEvent.model_validate(payload)
        topic = self._topics.build(
            _EVENT_ENVELOPE,
            site_id=self._site_id,
            zone_id=envelope.zone_id,
            modality=VERIFIED_EVENT_MODALITY,
        )
        await self._send(topic, envelope)
        return True

    async def publish_verified_events(self, envelopes: Sequence[EventEnvelope]) -> int:
        """Send accepted envelopes on the connected client, or open a one-shot client.

        A one-shot connection omits ``COOP_MQTT_CLIENT_ID`` so it does not take
        the session the API publisher already holds.

        Args:
            envelopes: Envelopes in ledger order. An empty sequence does not
                open a connection.

        Returns:
            How many envelopes were passed to the client seam.
        """
        if not envelopes:
            return 0
        if self._publish is not None:
            return await self._send_verified(envelopes)
        return await self._send_verified_once(envelopes)

    async def _send_verified(self, envelopes: Sequence[EventEnvelope]) -> int:
        sent = 0
        for envelope in envelopes:
            if await self.publish_verified_event(envelope):
                sent += 1
        return sent

    async def _send_verified_once(self, envelopes: Sequence[EventEnvelope]) -> int:
        import aiomqtt

        kwargs = self._mqtt.client_kwargs()
        kwargs.pop("identifier", None)
        kwargs["timeout"] = _ONCE_TIMEOUT_SEC
        qos = self._topics.entry(_EVENT_ENVELOPE).qos
        async with aiomqtt.Client(**kwargs) as client:

            async def _publish(topic: str, payload: bytes) -> None:
                await client.publish(topic, payload, qos=qos)

            previous = self._publish
            self._publish = _publish
            try:
                return await self._send_verified(envelopes)
            finally:
                self._publish = previous

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


def deliver_verified_envelopes(envelopes: Sequence[EventEnvelope]) -> None:
    """Hand accepted envelopes to the video MQTT publisher.

    A broker that is down is logged and ignored. The caller still records the
    ledger, so a later replay of those event ids does not send them again.
    Calling this from a running event loop is skipped for the same reason: the
    worker and the file analyzer call it from synchronous code.

    Args:
        envelopes: Newly accepted envelopes. The caller omits ids already in
            the ledger.
    """
    if not envelopes:
        return
    mission_id = str((envelopes[0].payload or {}).get("mission_id") or "")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        logger.warning(
            "verified event MQTT delivery skipped mission=%s count=%d reason=running_event_loop",
            mission_id,
            len(envelopes),
        )
        return
    publisher = VideoContractPublisher()
    try:
        sent = asyncio.run(publisher.publish_verified_events(envelopes))
    except Exception as exc:
        logger.warning(
            "verified event MQTT delivery failed mission=%s count=%d error=%s",
            mission_id,
            len(envelopes),
            type(exc).__name__,
        )
        return
    logger.info(
        "verified events handed to MQTT mission=%s count=%d modality=%s",
        mission_id,
        sent,
        VERIFIED_EVENT_MODALITY,
    )
