"""Today's site-event classes serialize to the ss-common golden fixtures.

Each case builds a message through the current code path (decoders, aggregator, realtime
conversions, API schema, captioner row), serializes it the way a publisher does, and asserts
that the result validates against the ODCS-generated model, emits no field outside the
contract, and equals the committed golden fixture under `tests/assets/contracts/golden/`
(snapshot of ss-common tag `v0.2.1`). The reverse direction rebuilds today's class from the
fixture. `SS_UPDATE_GOLDEN=1` rewrites the fixtures from the current code.
"""

import asyncio
import base64
import dataclasses
import json
import os
import pathlib
from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from selfsuvis.fusion_rt import sensor_ingest
from selfsuvis.fusion_rt.events import SensorEvent, ThreatEvent
from selfsuvis.fusion_rt.routers.v1.schemas import EventEnvelope
from selfsuvis.pipeline.media.rtsp_captioner import RtspCaptioner
from selfsuvis.pipeline.realtime.camera_events import CameraEvent, FrigateEventConsumer
from selfsuvis.pipeline.realtime.sound_analyzer import AcousticObservation, SoundAnalyzer
from ss_contracts.models import CONTRACTS

REPO = pathlib.Path(__file__).resolve().parents[3]
GOLDEN = REPO / "tests" / "assets" / "contracts" / "golden"
UPDATE = os.environ.get("SS_UPDATE_GOLDEN") == "1"

INGEST_TIME = "2026-09-19T08:00:01.250000+00:00"

FRIGATE_NEW: dict[str, Any] = {
    "type": "new",
    "before": {},
    "after": {
        "id": "1789804800.512345-k3p9qz",
        "camera": "entrance",
        "label": "person",
        "score": 0.8125,
        "top_score": 0.84375,
        "start_time": 1789804800.5,
        "end_time": None,
        "has_snapshot": True,
        "has_clip": False,
        "region": {"x": 0.25, "y": 0.125, "width": 0.5, "height": 0.75},
    },
}

FRIGATE_END: dict[str, Any] = {
    "type": "end",
    "after": {
        "id": "1789804900.100000-a7c1de",
        "camera": "yard",
        "label": "fox",
        "score": 0.625,
        "top_score": 0.9375,
        "start_time": 1789804900.25,
        "end_time": 1789804912.75,
        "has_snapshot": True,
        "has_clip": True,
        "region": {},
    },
}

FIXED = datetime(2026, 9, 19, 8, 0, 0, 500000, tzinfo=timezone.utc)


# -- Builders: one per golden fixture, through the current code path ----------------------


def _camera(payload: dict[str, Any]) -> CameraEvent:
    event = FrigateEventConsumer.decode(payload)
    assert event is not None
    return event


def _acoustic(audio: np.ndarray, transcript: str | None) -> AcousticObservation:
    captured: list[AcousticObservation] = []

    async def collect(observation: AcousticObservation) -> None:
        captured.append(observation)

    analyzer = SoundAnalyzer("entrance", "rtsp://frigate:8554/entrance", on_observation=collect)
    analyzer._capture_audio_chunk = lambda: audio  # type: ignore[method-assign]
    analyzer._transcribe = lambda _audio: transcript  # type: ignore[method-assign]
    asyncio.run(analyzer._process_chunk())
    return dataclasses.replace(captured[0], recorded_at=FIXED)


def _tone(freq_hz: float, amplitude: int) -> np.ndarray:
    t = np.arange(16_000 * 4) / 16_000
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)


def _with_ingest_time(build: Callable[[], Any]) -> Any:
    original = sensor_ingest._now_iso
    sensor_ingest._now_iso = lambda: INGEST_TIME
    try:
        return build()
    finally:
        sensor_ingest._now_iso = original


def _scene_row(caption: dict[str, Any], gps: tuple[float | None, ...]) -> dict[str, Any]:
    class _Pool:
        args: tuple[Any, ...] = ()

        async def execute(self, _query: str, *args: Any) -> None:
            self.args = args

    pool = _Pool()
    captioner = RtspCaptioner("rtsp://localhost:8554/drone1", "mission-0917", pool, 1.0)
    asyncio.run(
        captioner._write_to_timeline(
            "frame-000042", caption["caption"], caption["facts_json"], *gps, 12.5
        )
    )
    columns = ("mission_id", "frame_id", "gps_lat", "gps_lon", "gps_alt", "t_sec", "caption")
    row = dict(zip(columns, pool.args[:7], strict=True))
    row["facts_json"] = json.loads(pool.args[7]) if pool.args[7] is not None else None
    row["created_at"] = FIXED
    return row


GEMMA_FACTS = {
    "scene_summary": "A red pickup truck parked beside the north gate.",
    "objects": ["truck", "gate"],
    "people_count": 0,
}

CASES: dict[tuple[str, str], Callable[[], Any]] = {
    ("camera-event", "new"): lambda: _camera(FRIGATE_NEW),
    ("camera-event", "end"): lambda: _camera(FRIGATE_END),
    ("acoustic-observation", "alarm"): lambda: _acoustic(_tone(3000.0, 12000), "stay back"),
    ("acoustic-observation", "silent"): lambda: _acoustic(_tone(440.0, 1), None),
    ("threat-event", "camera"): lambda: _with_ingest_time(
        lambda: sensor_ingest.camera_event_to_threat(_camera(FRIGATE_NEW))
    ),
    ("event-envelope", "audio"): lambda: EventEnvelope(
        ts="2026-09-19T08:02:03.5+00:00",
        zone_id="north-gate",
        sensor_id="mic-03",
        confidence=0.91,
        payload={"label": "drone", "band_hz": [120, 380]},
        artifact_uri="clips/north-gate/2026-09-19T080203.wav",
    ),
    ("event-envelope", "minimal"): lambda: EventEnvelope(
        ts="2026-09-19T08:05:00Z", zone_id="yard", sensor_id="pir-7", confidence=1.0
    ),
    ("scene-caption", "gemma"): lambda: _scene_row(
        {"caption": GEMMA_FACTS["scene_summary"], "facts_json": GEMMA_FACTS},
        (50.4502, 30.5236, 181.5),
    ),
    ("scene-caption", "florence-no-gps"): lambda: _scene_row(
        {"caption": "a dirt road next to a field", "facts_json": None}, (None, None, None)
    ),
}


# -- Serialization, as a publisher writes the message ------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def to_wire(obj: Any) -> dict[str, Any]:
    """Serialize today's object to a contract message; unset optional fields are omitted."""
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json")
    elif hasattr(obj, "to_dict"):
        data = obj.to_dict()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        data = dataclasses.asdict(obj)
    else:
        data = dict(obj)
    wire: dict[str, Any] = json.loads(json.dumps(data, default=_json_default))
    return {key: value for key, value in wire.items() if value is not None}


def assert_matches(actual: Any, expected: Any) -> None:
    """Exact equality, except floats (numpy-computed levels may differ in the last ulp)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), (actual, expected)
        for key, value in expected.items():
            assert_matches(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), (actual, expected)
        for item, value in zip(actual, expected, strict=True):
            assert_matches(item, value)
    elif isinstance(expected, float) and not isinstance(actual, bool):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert type(actual) is type(expected) and actual == expected, (actual, expected)


def _canonical(contract_id: str, wire: dict[str, Any]) -> dict[str, Any]:
    model = CONTRACTS[contract_id]
    emitted = set(wire) - set(model.model_fields)
    assert not emitted, f"{contract_id}: fields outside the contract: {sorted(emitted)}"
    return model.model_validate(wire).model_dump(mode="json", exclude_unset=True)


def _ids(cases: list[tuple[str, str]]) -> list[str]:
    return [f"{contract_id}/{name}" for contract_id, name in cases]


@pytest.mark.parametrize(("contract_id", "name"), sorted(CASES), ids=_ids(sorted(CASES)))
def test_current_class_serializes_to_golden(contract_id: str, name: str) -> None:
    wire = to_wire(CASES[contract_id, name]())
    canonical = _canonical(contract_id, wire)
    path = GOLDEN / contract_id / f"{name}.json"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
    assert_matches(canonical, json.loads(path.read_text(encoding="utf-8")))


REBUILD: dict[str, Callable[[dict[str, Any]], Any]] = {
    "camera-event": lambda fields: CameraEvent(**fields),
    "acoustic-observation": lambda fields: AcousticObservation(**fields),
    "sensor-event": lambda fields: SensorEvent(**_without(fields, "event_kind")),
    "threat-event": lambda fields: ThreatEvent(**_without(fields, "event_kind")),
    # Pydantic classes parse the JSON form: EventEnvelope.ts is a string today.
    "event-envelope": EventEnvelope.model_validate_json,
}
PARSES_JSON = {"event-envelope"}

# Dataclass fields without defaults that the wire form omits when unset.
NULLABLE_WITHOUT_DEFAULT = {
    "camera-event": ("ended_at",),
    "acoustic-observation": ("speech_transcript",),
}


def _without(fields: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in fields.items() if name != key}


REBUILD_CASES = sorted(case for case in CASES if case[0] in REBUILD)
REBUILD_CASES.extend(
    sorted(
        (path.parent.name, path.stem)
        for path in GOLDEN.glob("sensor-event/*.json")
        if not path.name.endswith(".invalid.json")
    )
)


@pytest.mark.parametrize(("contract_id", "name"), REBUILD_CASES, ids=_ids(REBUILD_CASES))
def test_golden_rebuilds_current_class(contract_id: str, name: str) -> None:
    golden = json.loads((GOLDEN / contract_id / f"{name}.json").read_text(encoding="utf-8"))
    message = CONTRACTS[contract_id].model_validate(golden)
    fields = message.model_dump(mode="python", exclude_unset=True)
    for key in NULLABLE_WITHOUT_DEFAULT.get(contract_id, ()):
        fields.setdefault(key, None)
    rebuilt = REBUILD[contract_id](
        message.model_dump_json(exclude_unset=True) if contract_id in PARSES_JSON else fields
    )
    assert_matches(_canonical(contract_id, to_wire(rebuilt)), golden)


def test_every_contract_has_a_captured_fixture() -> None:
    # Site events are the topic-bound contracts; file manifests are covered by
    # test_manifest_contracts.py. Sensor-reading/state/event goldens are produced
    # by ss-sens (`tests/unit/test_site_event_contracts.py` in volod/ss-sens).
    topic_bound = {cid for cid, model in CONTRACTS.items() if model.SS_BINDING.get("mqttTopic")}
    golden_ids = {path.name for path in GOLDEN.iterdir() if path.is_dir()}
    assert topic_bound <= golden_ids
    video_produced = {
        "camera-event",
        "acoustic-observation",
        "threat-event",
        "event-envelope",
        "scene-caption",
    }
    assert {contract_id for contract_id, _name in CASES} == video_produced
