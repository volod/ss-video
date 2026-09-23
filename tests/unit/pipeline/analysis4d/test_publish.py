"""Accepted 4D events are handed to the video MQTT publisher once."""

from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import yaml

from selfsuvis.pipeline.analysis4d.publish import publish_accepted
from selfsuvis.pipeline.analysis4d.schemas import SceneTimeline, Verification
from selfsuvis.pipeline.realtime.contract_publisher import VideoContractPublisher

_WHEN = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _event(event_id: str, status: str, *, start: float) -> dict:
    evidence = []
    if status == "accepted":
        evidence = [
            {
                "frame_id": f"mission-delivery:{event_id}:0000000",
                "t_sec": start,
                "track_ids": ["track-1"],
            }
        ]
    return {
        "event_id": event_id,
        "type": "count_changed",
        "summary": status,
        "start_sec": start,
        "end_sec": start + 1.0,
        "participants": ["track-1"],
        "confidence": 0.9 if status == "accepted" else 0.2,
        "verification": Verification(status=status, rules=["count"], reasons=[]),
        "evidence": evidence,
    }


def _timeline(*events: dict) -> SceneTimeline:
    return SceneTimeline(
        schema_version="ss-video.scene-timeline.v1",
        mission_id="mission-delivery",
        profile="fast",
        coordinate_frame={"name": "mission_enu", "metric_scale": "unavailable"},
        model_manifest_ref="manifest.json",
        events=list(events),
    )


def _patch_publish(monkeypatch, calls: list[list[str]]) -> None:
    async def _fake(self, envelopes) -> int:
        del self
        calls.append([item.payload["event_id"] for item in envelopes])
        return len(envelopes)

    monkeypatch.setattr(VideoContractPublisher, "publish_verified_events", _fake)


def test_repeated_event_id_produces_one_publish_call(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []
    _patch_publish(monkeypatch, calls)
    timeline = _timeline(_event("evt-keep", "accepted", start=0.0))
    kwargs = {
        "zone_id": "yard",
        "sensor_id": "cam",
        "published_at": _WHEN,
    }
    assert publish_accepted(tmp_path, timeline, **kwargs) == ["evt-keep"]
    assert publish_accepted(tmp_path, timeline, **kwargs) == []
    assert calls == [["evt-keep"]]
    ledger = (tmp_path / "published-events.jsonl").read_text(encoding="utf-8")
    assert ledger.count("evt-keep") == 1


def test_rejected_and_uncertain_rows_are_not_published(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []
    _patch_publish(monkeypatch, calls)
    timeline = _timeline(
        _event("evt-keep", "accepted", start=0.0),
        _event("evt-reject", "rejected", start=1.0),
        _event("evt-unsure", "uncertain", start=2.0),
    )
    fresh = publish_accepted(
        tmp_path,
        timeline,
        zone_id="yard",
        sensor_id="cam",
        published_at=_WHEN,
    )
    assert fresh == ["evt-keep"]
    assert calls == [["evt-keep"]]
    text = (tmp_path / "published-events.jsonl").read_text(encoding="utf-8")
    assert "evt-reject" not in text
    assert "evt-unsure" not in text

    empty_calls: list[list[str]] = []
    _patch_publish(monkeypatch, empty_calls)
    dropped = _timeline(
        _event("evt-reject", "rejected", start=1.0),
        _event("evt-unsure", "uncertain", start=2.0),
    )
    assert publish_accepted(tmp_path, dropped, zone_id="yard", sensor_id="cam") == []
    assert empty_calls == []


def test_unreachable_broker_still_writes_the_ledger(tmp_path: Path, monkeypatch) -> None:
    async def _refused(self, envelopes) -> int:
        del self, envelopes
        raise ConnectionRefusedError("broker refused")

    monkeypatch.setattr(VideoContractPublisher, "publish_verified_events", _refused)
    timeline = _timeline(_event("evt-keep", "accepted", start=0.0))
    fresh = publish_accepted(
        tmp_path,
        timeline,
        zone_id="yard",
        sensor_id="cam",
        published_at=_WHEN,
    )
    assert fresh == ["evt-keep"]
    assert "evt-keep" in (tmp_path / "published-events.jsonl").read_text(encoding="utf-8")
    again = publish_accepted(
        tmp_path,
        timeline,
        zone_id="yard",
        sensor_id="cam",
        published_at=_WHEN,
    )
    assert again == []


def test_fusion_rule_seed_is_unchanged() -> None:
    text = (
        resources.files("selfsuvis.fusion_rt")
        .joinpath("data", "fusion_rules.yaml")
        .read_text(encoding="utf-8")
    )
    loaded = yaml.safe_load(text)
    assert loaded["rules"] == []
