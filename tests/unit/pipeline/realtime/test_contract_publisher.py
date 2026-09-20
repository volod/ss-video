"""VideoContractPublisher serializes Frigate events onto the topic map."""

from datetime import datetime, timezone

from selfsuvis.pipeline.realtime.camera_events import CameraEvent
from selfsuvis.pipeline.realtime.contract_publisher import (
    camera_event_to_contract,
    scene_caption_to_contract,
)


def test_camera_event_to_contract_round_trip() -> None:
    event = CameraEvent(
        event_id="e1",
        camera="entrance",
        label="person",
        score=0.81,
        top_score=0.84,
        event_type="new",
        started_at=datetime(2026, 9, 19, 8, 0, tzinfo=timezone.utc),
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
        created_at=datetime(2026, 9, 19, 8, 0, tzinfo=timezone.utc),
    )
    assert model.mission_id == "mission-1"
    assert model.caption == "a truck at the gate"
    assert model.t_sec == 1.5
