import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from selfsuvis.fusion_rt import (
    NodeHealthEvent,
    RealtimeThreatAggregator,
    SensorEvent,
    ThreatEvent,
    evaluate_degraded_mode,
)
from selfsuvis.pipeline.realtime import (
    freshness_seconds,
    load_jsonl_records,
    replay_bridge_trace,
    replay_local_run,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_event_envelopes_normalize_core_fields():
    event = SensorEvent(
        event_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ingest_time="2026-01-01T00:00:05Z",
        node_id=" Node_A ",
        sensor_type=" Camera ",
        sector_id=" sector_1 ",
        payload={"frame_id": "f1"},
    ).to_dict()
    assert event["event_kind"] == "sensor"
    assert event["node_id"] == "node_a"
    assert event["sensor_type"] == "camera"
    assert event["sector_id"] == "sector_1"
    assert freshness_seconds(event["event_time"], event["ingest_time"]) == 5.0


def test_replay_local_run_produces_ordered_events_with_freshness():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        video_dir = output_dir / "video_a"
        _write_json(
            video_dir / "local_threat_assessment.json",
            {
                "local_threat_score": 0.62,
                "automation_confidence": 0.58,
                "trust_penalty": 0.12,
                "recommended_action": "reduce_speed",
                "top_threats": [
                    {
                        "type": "collision_risk",
                        "score": 0.62,
                        "evidence": {
                            "evidence_sources": ["near_field_occupancy", "object_velocity"]
                        },
                    }
                ],
            },
        )
        _write_json(
            video_dir / "threat_primitives.json",
            {
                "primitives": [
                    {
                        "type": "collision_risk",
                        "score": 0.62,
                        "uncertainty": 0.15,
                        "evidence_sources": ["near_field_occupancy", "object_velocity"],
                    },
                    {
                        "type": "visibility_degradation",
                        "score": 0.44,
                        "uncertainty": 0.20,
                        "evidence_sources": ["depth_failure_rate", "caption_confidence"],
                    },
                ]
            },
        )
        _write_json(video_dir / "physical_state_summary.json", {"platform_pose_confidence": 0.77})
        _write_json(
            video_dir / "field_state_summary.json",
            {"clip_level_fields": {"visibility": {"mean": 0.22}}},
        )
        _write_json(
            video_dir / "full_state_fusion.json",
            {
                "platform": {"origin_lla": {"lat": 50.4501, "lon": 30.5234, "alt": 100.0}},
                "map_state": {
                    "smoothed_samples": [
                        {"t_sec": 0.0, "position_enu_m": {"x": 0.0, "y": 0.0, "z": 0.0}},
                        {"t_sec": 1.5, "position_enu_m": {"x": 25.0, "y": 5.0, "z": 0.0}},
                    ]
                },
            },
        )

        events = replay_local_run(
            output_dir,
            ingest_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ingest_delay_sec_by_kind={"threat": 4.0},
        )

        assert len(events) == 5
        assert events == sorted(
            events,
            key=lambda row: (
                row["event_time"],
                row["event_kind"],
                row["sector_id"],
                row["sensor_type"],
            ),
        )
        assert any(
            event["event_kind"] == "threat" and event["freshness_sec"] >= 4.0 for event in events
        )


def test_degraded_mode_and_aggregator_reflect_stale_and_missing_inputs():
    sensor_events = [
        SensorEvent(
            event_time="2026-01-01T00:00:00+00:00",
            ingest_time="2026-01-01T00:00:45+00:00",
            node_id="node_a",
            sensor_type="camera",
            sector_id="sector_1",
            payload={"frame_id": "f1"},
            freshness_sec=45.0,
        ).to_dict(),
    ]
    health_events = [
        NodeHealthEvent(
            event_time="2026-01-01T00:00:00+00:00",
            ingest_time="2026-01-01T00:00:45+00:00",
            node_id="node_a",
            sensor_type="analytics",
            sector_id="sector_1",
            payload={"automation_confidence": 0.74, "outage_sec": 50.0, "model_name": "unidrive"},
            freshness_sec=45.0,
        ).to_dict(),
    ]
    degraded = evaluate_degraded_mode(
        sensor_events,
        health_events,
        base_automation_confidence=0.74,
    )
    assert degraded["degraded"] is True
    assert degraded["automation_confidence"] < 0.74
    assert degraded["health_warnings"]

    aggregator = RealtimeThreatAggregator()
    aggregator.consume_all(
        sensor_events
        + health_events
        + [
            ThreatEvent(
                event_time="2026-01-01T00:00:00+00:00",
                ingest_time="2026-01-01T00:00:45+00:00",
                node_id="node_a",
                sensor_type="local_threat",
                sector_id="sector_1",
                payload={
                    "route_id": "route_a",
                    "local_threat_score": 0.66,
                    "automation_confidence": 0.74,
                },
                freshness_sec=45.0,
            ).to_dict()
        ]
    )
    snapshot = aggregator.snapshot()
    assert snapshot["automation_confidence"] < 0.74
    assert snapshot["degraded"] is True
    assert snapshot["route_advisories"][0]["recommended_action"] == "inspect_sensor"


def test_replay_bridge_trace_loads_recorded_mavlink_and_ros_samples():
    fixture_dir = Path(__file__).resolve().parents[3] / "fixtures" / "realtime"

    mav_records = load_jsonl_records(fixture_dir / "mavlink_trace.jsonl")
    ros_records = load_jsonl_records(fixture_dir / "ros_trace.jsonl")
    mav_packets = replay_bridge_trace(fixture_dir / "mavlink_trace.jsonl", backend="mavlink")
    ros_packets = replay_bridge_trace(fixture_dir / "ros_trace.jsonl", backend="ros")

    assert len(mav_records) == 4
    assert len(ros_records) == 5
    assert [packet["sensor_type"] for packet in mav_packets] == [
        "gps",
        "imu",
        "barometer",
        "magnetometer",
    ]
    assert [packet["sensor_type"] for packet in ros_packets] == [
        "gps",
        "imu",
        "barometer",
        "magnetometer",
        "camera",
    ]
