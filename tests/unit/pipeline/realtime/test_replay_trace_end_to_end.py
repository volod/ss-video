from pathlib import Path

import pytest
from tests.support.realtime_db import FakeRealtimeConn

from selfsuvis.app.services.realtime import ingest_realtime_packets, start_realtime_session
from selfsuvis.pipeline.realtime import replay_bridge_trace
from selfsuvis.pipeline.storage.realtime import fetch_realtime_state, summarize_realtime_session


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("backend", "fixture_name", "expected_pose_source", "expected_packet_counts"),
    [
        (
            "mavlink",
            "mavlink_trace.jsonl",
            "fused_gps_imu_barometer",
            {"barometer": 1, "gps": 1, "imu": 1, "magnetometer": 1},
        ),
        (
            "ros",
            "ros_trace.jsonl",
            "fused_gps_imu_barometer",
            {"barometer": 1, "camera": 1, "gps": 1, "imu": 1, "magnetometer": 1},
        ),
    ],
)
async def test_replay_bridge_trace_end_to_end_ingests_recorded_samples(
    backend: str,
    fixture_name: str,
    expected_pose_source: str,
    expected_packet_counts: dict[str, int],
):
    fixture_dir = Path(__file__).resolve().parents[3] / "fixtures" / "realtime"
    conn = FakeRealtimeConn()
    session = await start_realtime_session(
        conn,
        robot_id=f"robot_{backend}",
        mission_id=f"mission_{backend}",
        sensors=["camera", "gps", "imu", "barometer", "magnetometer"],
    )
    packets = replay_bridge_trace(fixture_dir / fixture_name, backend=backend)

    result = await ingest_realtime_packets(conn, session_id=session["session_id"], packets=packets)
    state = await fetch_realtime_state(conn, session["session_id"])
    summary = await summarize_realtime_session(conn, session["session_id"])

    assert result["accepted_packets"] == len(packets)
    assert result["pose_updated"] is True
    assert state is not None
    assert state["packet_counts"] == expected_packet_counts
    assert state["latest_pose"]["source"] == expected_pose_source
    assert state["latest_pose"]["tracking_status"] == "ok"
    assert summary["duration_sec"] >= 0.0
