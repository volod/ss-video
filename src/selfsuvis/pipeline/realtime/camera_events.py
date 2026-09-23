"""Consume Frigate NVR MQTT events into typed CameraEvent objects.

Frigate publishes on:
  {prefix}/events                — new/update/end events for all cameras
  {prefix}/{camera_name}/events  — camera-specific events

Event payload shape (Frigate ≥ 0.13):
  {
    "before": {...},
    "after": {
      "id": "...", "camera": "front_door", "label": "person",
      "score": 0.87, "top_score": 0.92,
      "start_time": 1712000000.0, "end_time": null,
      "has_snapshot": true, "has_clip": false,
      "area": 12345, "ratio": 1.8,
      "region": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
      ...
    },
    "type": "new"  # "new" | "update" | "end"
  }
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class CameraEvent:
    """A detection event from Frigate NVR."""

    event_id: str
    camera: str
    label: str
    score: float
    top_score: float
    event_type: str  # "new" | "update" | "end"
    started_at: datetime
    ended_at: datetime | None
    has_snapshot: bool
    has_clip: bool
    region: dict[str, float]  # {x, y, width, height} normalized 0-1
    raw: dict[str, Any]


class FrigateEventConsumer:
    """Decode raw Frigate MQTT payloads into CameraEvent objects."""

    @staticmethod
    def decode(payload: str | bytes | dict) -> CameraEvent | None:
        """Parse a Frigate event MQTT payload.

        Returns None if the payload cannot be parsed or has no `after` state.
        """
        try:
            if isinstance(payload, dict):
                msg = payload
            else:
                msg = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None

        after: dict[str, Any] = msg.get("after") or {}
        if not after:
            return None

        try:
            started_at = datetime.fromtimestamp(after["start_time"], tz=UTC)
        except (KeyError, TypeError, ValueError, OSError):
            started_at = datetime.now(UTC)

        ended_at: datetime | None = None
        if (end_ts := after.get("end_time")) is not None:
            try:
                ended_at = datetime.fromtimestamp(end_ts, tz=UTC)
            except (TypeError, ValueError, OSError):
                pass

        # Only the mapping form is decoded; a box list or other shape yields an empty region.
        region = after.get("region")
        if not isinstance(region, dict):
            region = {}

        return CameraEvent(
            event_id=after.get("id", ""),
            camera=after.get("camera", "unknown"),
            label=after.get("label", "unknown"),
            score=_to_float(after.get("score")),
            top_score=_to_float(after.get("top_score")),
            event_type=msg.get("type", "update"),
            started_at=started_at,
            ended_at=ended_at,
            has_snapshot=bool(after.get("has_snapshot")),
            has_clip=bool(after.get("has_clip")),
            region={
                k: parsed
                for k, v in region.items()
                if (parsed := _to_optional_float(v)) is not None
            },
            raw=msg,
        )


def _to_float(value: Any) -> float:
    return _to_optional_float(value) or 0.0


def _to_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
