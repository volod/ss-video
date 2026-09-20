"""Realtime semantic-observation helpers."""

from typing import Any


def normalize_semantic_observation(observation: dict[str, Any]) -> dict[str, Any]:
    class_name = str(observation.get("class_name", "")).strip().lower()
    if not class_name:
        raise ValueError("class_name is required")
    confidence = float(observation.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be within [0, 1]")
    return {
        "frame_id": observation.get("frame_id"),
        "class_name": class_name,
        "confidence": confidence,
        "position_enu": dict(observation.get("position_enu") or {}) or None,
        "bbox": dict(observation.get("bbox") or {}) or None,
        "mask_ref": observation.get("mask_ref"),
        "track_id": observation.get("track_id"),
        "facts": dict(observation.get("facts") or {}),
    }


def project_detection_to_enu(
    *,
    pose: dict[str, Any],
    bbox: dict[str, Any],
    range_m: float | None = None,
) -> dict[str, float] | None:
    """Approximate ENU backprojection for a detection.

    This is intentionally lightweight: it uses the bbox center as an angular hint
    and projects it forward from the current ENU pose with a coarse range.
    """
    position = dict(pose.get("position_enu") or {})
    if "x" not in position or "y" not in position:
        return None
    x1 = float(bbox.get("x1", bbox.get("xmin", 0.0)) or 0.0)
    x2 = float(bbox.get("x2", bbox.get("xmax", 1.0)) or 1.0)
    y1 = float(bbox.get("y1", bbox.get("ymin", 0.0)) or 0.0)
    y2 = float(bbox.get("y2", bbox.get("ymax", 1.0)) or 1.0)
    cx = max(0.0, min(1.0, (x1 + x2) / 2.0))
    cy = max(0.0, min(1.0, (y1 + y2) / 2.0))
    distance = float(
        range_m if range_m is not None else bbox.get("depth_m", bbox.get("range_m", 10.0)) or 10.0
    )
    east_offset = (cx - 0.5) * distance
    north_offset = (0.5 - cy) * distance
    return {
        "x": float(position["x"]) + east_offset,
        "y": float(position["y"]) + north_offset,
        "z": float(position.get("z", 0.0) or 0.0),
    }
