"""Focused realtime sync / lag-bound tests."""

from selfsuvis.pipeline.realtime.pose import build_fused_pose_from_packets


def test_pose_sync_drops_when_gps_anchor_is_too_stale():
    pose = build_fused_pose_from_packets(
        [
            {"sensor_type": "gps", "t_device": 1.0, "payload": {"east": 1.0, "north": 2.0}},
            {"sensor_type": "imu", "t_device": 1.4, "payload": {"yaw": 0.1}},
        ],
        max_lag_ms=100,
    )
    assert pose is None


def test_pose_sync_keeps_close_packets():
    pose = build_fused_pose_from_packets(
        [
            {"sensor_type": "gps", "t_device": 1.0, "payload": {"east": 1.0, "north": 2.0}},
            {"sensor_type": "imu", "t_device": 1.05, "payload": {"yaw": 0.1}},
        ],
        max_lag_ms=100,
    )
    assert pose is not None
    assert pose["source"] == "fused_gps_imu"
