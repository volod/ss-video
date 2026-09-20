"""Focused realtime ingest tests."""

import pytest

from selfsuvis.pipeline.realtime.ingest import normalize_packets


def test_normalize_packets_accepts_camera_and_lidar():
    packets = normalize_packets(
        [
            {"sensor_type": "camera", "t_device": 1.0, "payload": {"frame_id": "f1"}},
            {"sensor_type": "LiDAR", "t_device": 1.1, "payload": {"points": 42}},
        ]
    )
    assert packets[0]["sensor_type"] == "camera"
    assert packets[1]["sensor_type"] == "lidar"


def test_normalize_packets_rejects_missing_time():
    with pytest.raises(ValueError, match="packet missing t_device"):
        normalize_packets([{"sensor_type": "camera"}])
