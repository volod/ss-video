"""Semantic detection to ENU backprojection tests."""

from selfsuvis.pipeline.realtime.semantics import project_detection_to_enu


def test_project_detection_to_enu_uses_bbox_center_offset():
    pose = {"position_enu": {"x": 10.0, "y": 20.0, "z": 2.0}}
    bbox = {"x1": 0.75, "x2": 0.95, "y1": 0.25, "y2": 0.45}
    projected = project_detection_to_enu(pose=pose, bbox=bbox, range_m=8.0)

    assert projected is not None
    assert projected["x"] > 10.0
    assert projected["y"] > 20.0
    assert projected["z"] == 2.0
