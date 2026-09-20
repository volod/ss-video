"""Frigate decoder and acoustic observation scalars (video-side camera code)."""

import json
from pathlib import Path

from selfsuvis.pipeline.realtime.camera_events import FrigateEventConsumer

REPO = Path(__file__).resolve().parents[4]
FRIGATE_BOX = REPO / "tests" / "assets" / "frigate" / "event_box_list.json"


def test_frigate_event_decode_handles_numeric_strings_and_bad_values() -> None:
    event = FrigateEventConsumer.decode(
        {
            "type": "new",
            "after": {
                "id": "event-1",
                "camera": "entrance",
                "label": "person",
                "score": "0.88",
                "top_score": "bad",
                "start_time": 1777622400.0,
                "end_time": 0,
                "has_snapshot": True,
                "has_clip": False,
                "region": {"x": "0.1", "y": "bad", "width": 0.3},
            },
        }
    )

    assert event is not None
    assert event.score == 0.88
    assert event.top_score == 0.0
    assert event.ended_at is not None
    assert event.region == {"x": 0.1, "width": 0.3}


def test_frigate_event_decode_tolerates_a_non_mapping_region() -> None:
    event = FrigateEventConsumer.decode(
        {
            "type": "update",
            "after": {
                "id": "event-2",
                "camera": "yard",
                "label": "car",
                "score": 0.7,
                "start_time": 1777622400.0,
                "region": [264, 450, 667, 853],
            },
        }
    )

    assert event is not None
    assert event.region == {}


def test_frigate_fixture_event_with_box_list_decodes() -> None:
    payload = json.loads(FRIGATE_BOX.read_text(encoding="utf-8"))
    event = FrigateEventConsumer.decode(payload)
    assert event is not None
    assert event.camera == "entrance"
    assert event.label == "person"
    assert event.region == {}


def test_sound_analyzer_emits_plain_python_scalars() -> None:
    import asyncio

    import numpy as np

    from selfsuvis.pipeline.realtime.sound_analyzer import SoundAnalyzer

    observations = []

    async def collect(observation) -> None:
        observations.append(observation)

    analyzer = SoundAnalyzer("entrance", "rtsp://frigate:8554/entrance", on_observation=collect)
    analyzer._capture_audio_chunk = lambda: np.full(16_000, 1000, dtype=np.int16)
    analyzer._transcribe = lambda _audio: None
    asyncio.run(analyzer._process_chunk())

    observation = observations[0]
    assert type(observation.silence) is bool
    assert type(observation.rms_db) is float
    json.dumps({"silence": observation.silence, "rms_db": observation.rms_db})
