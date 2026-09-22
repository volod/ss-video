"""VideoContractPublisher serializes Frigate events onto the topic map."""

import json
from datetime import UTC, datetime

from selfsuvis.pipeline.analysis4d.verified_scene_event import VerifiedSceneEvent
from selfsuvis.pipeline.realtime.camera_events import CameraEvent
from selfsuvis.pipeline.realtime.contract_publisher import (
    VERIFIED_EVENT_MODALITY,
    VideoContractPublisher,
    camera_event_to_contract,
    scene_caption_to_contract,
)
from ss_contracts.models import EventEnvelope
from ss_kit.mqtt import MqttSettings


def test_camera_event_to_contract_round_trip() -> None:
    event = CameraEvent(
        event_id="e1",
        camera="entrance",
        label="person",
        score=0.81,
        top_score=0.84,
        event_type="new",
        started_at=datetime(2026, 9, 19, 8, 0, tzinfo=UTC),
        ended_at=None,
        has_snapshot=True,
        has_clip=False,
        region={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
        raw={"type": "new"},
    )
    model = camera_event_to_contract(event)
    assert model.camera == "entrance"
    assert model.label == "person"
    dumped = model.model_dump(mode="json")
    assert dumped["event_id"] == "e1"


def test_scene_caption_to_contract() -> None:
    model = scene_caption_to_contract(
        mission_id="mission-1",
        frame_id="frame-1",
        caption="a truck at the gate",
        facts_json={"objects": ["truck"]},
        gps_lat=None,
        gps_lon=None,
        gps_alt=None,
        t_sec=1.5,
        created_at=datetime(2026, 9, 19, 8, 0, tzinfo=UTC),
    )
    assert model.mission_id == "mission-1"
    assert model.caption == "a truck at the gate"
    assert model.t_sec == 1.5


def _verified_envelope(status: str = "accepted") -> EventEnvelope:
    stamp = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
    if status == "accepted":
        payload = VerifiedSceneEvent(
            event_id="evt-keep",
            mission_id="mission-1",
            profile="fast",
            event_type="count_changed",
            summary="kept",
            start_sec=0.0,
            end_sec=1.0,
            participants=["track-1"],
            confidence=0.9,
            verification_status="accepted",
            artifact_uri="timeline.json",
            published_at=stamp,
        ).model_dump(mode="json")
    else:
        payload = {
            "event_id": "evt-drop",
            "verification_status": status,
        }
    return EventEnvelope(
        ts=stamp,
        zone_id="yard",
        sensor_id="cam",
        confidence=0.9,
        payload=payload,
        artifact_uri="timeline.json",
    )


async def test_verified_event_uses_the_mqtt_client_seam() -> None:
    assert VERIFIED_EVENT_MODALITY == VerifiedSceneEvent.SS_BINDING["modality"]
    publisher = VideoContractPublisher(site_id="site-a")
    sent: list[tuple[str, bytes]] = []

    async def _publish(topic: str, payload: bytes) -> None:
        sent.append((topic, payload))

    publisher._publish = _publish
    sent_count = await publisher.publish_verified_events(
        [_verified_envelope(), _verified_envelope("rejected"), _verified_envelope("uncertain")]
    )
    assert sent_count == 1
    assert sent[0][0] == "ss/v1/site/site-a/zone/yard/event/video_4d"
    body = json.loads(sent[0][1])
    EventEnvelope.model_validate(body)
    payload = VerifiedSceneEvent.model_validate(body["payload"])
    assert payload.event_id == "evt-keep"
    assert payload.verification_status == "accepted"


async def test_one_shot_client_publishes_without_the_api_client_id(monkeypatch) -> None:
    sent: list[tuple[str, bytes, int, dict]] = []

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        async def publish(self, topic: str, payload: bytes, qos: int = 0) -> None:
            sent.append((topic, payload, qos, self.kwargs))

    monkeypatch.setattr("aiomqtt.Client", _Client)
    publisher = VideoContractPublisher(
        mqtt=MqttSettings(host="127.0.0.1", port=1, client_id="video-api"),
        site_id="site-a",
    )
    assert await publisher.publish_verified_events([_verified_envelope()]) == 1
    topic, payload, qos, kwargs = sent[0]
    assert topic == "ss/v1/site/site-a/zone/yard/event/video_4d"
    assert qos == 1
    assert kwargs["hostname"] == "127.0.0.1"
    assert kwargs["timeout"] == 2.0
    assert "identifier" not in kwargs
    assert VerifiedSceneEvent.model_validate(json.loads(payload)["payload"]).event_id == "evt-keep"
