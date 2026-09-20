"""ContractEventConsumer dispatches sensor-event and sensor-state payloads."""

import json
from datetime import datetime, timezone

from selfsuvis.pipeline.realtime.contract_consumer import ContractEventConsumer
from ss_kit.mqtt import TopicBuilder


async def test_handle_message_dispatches_sensor_event_and_state() -> None:
    events: list[object] = []
    states: list[object] = []

    async def on_event(model: object) -> None:
        events.append(model)

    async def on_state(model: object) -> None:
        states.append(model)

    consumer = ContractEventConsumer(
        on_sensor_event=on_event,
        on_sensor_state=on_state,
        subscribe_frigate=False,
    )
    topics = TopicBuilder()
    event_topic = topics.build("sensor-event", site_id="local", node_id="70b3d57ed0060001")
    state_topic = topics.build("sensor-state", site_id="local", dev_eui="70b3d57ed0060001")
    event_body = {
        "event_kind": "sensor",
        "event_time": "2026-09-19T07:59:58.412345Z",
        "ingest_time": "2026-09-19T08:00:01.250000Z",
        "node_id": "70b3d57ed0060001",
        "sensor_type": "lorawan",
        "sector_id": "grid:50450:30523",
        "payload": {"temperature_c": 21.4},
        "freshness_sec": 0.0,
    }
    state_body = {
        "dev_eui": "70b3d57ed0060001",
        "last_seen": datetime(2026, 9, 19, 8, 0, tzinfo=timezone.utc).isoformat(),
        "reading_count": 2,
        "temperature_c": 21.4,
    }
    await consumer.handle_message(event_topic, json.dumps(event_body).encode("utf-8"))
    await consumer.handle_message(state_topic, json.dumps(state_body).encode("utf-8"))
    assert len(events) == 1
    assert getattr(events[0], "node_id") == "70b3d57ed0060001"
    assert len(states) == 1
    assert getattr(states[0], "reading_count") == 2


async def test_handle_message_ignores_invalid_payload() -> None:
    seen: list[object] = []

    async def on_event(model: object) -> None:
        seen.append(model)

    consumer = ContractEventConsumer(on_sensor_event=on_event, subscribe_frigate=False)
    topic = TopicBuilder().build("sensor-event", site_id="local", node_id="x")
    await consumer.handle_message(topic, b"not-json")
    assert seen == []
