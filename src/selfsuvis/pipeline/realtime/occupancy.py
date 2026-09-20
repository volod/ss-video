"""Realtime occupancy/tile helpers."""

import json
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core import ensure_dir, settings
from selfsuvis.pipeline.realtime.sidecar import RealtimeSidecarClient
from selfsuvis.realtime.adapters import create_occupancy_adapter


def normalize_map_tile(tile: dict[str, Any]) -> dict[str, Any]:
    return {
        "tile_key": str(tile["tile_key"]),
        "map_type": str(tile.get("map_type", "occupancy")).strip().lower(),
        "storage_path": str(tile["storage_path"]),
        "resolution_m": float(tile.get("resolution_m", 0.2)),
        "bounds": dict(tile.get("bounds") or {}),
        "stats": dict(tile.get("stats") or {}),
        "global_map_id": int(tile["global_map_id"])
        if tile.get("global_map_id") is not None
        else None,
    }


def realtime_tile_dir(session_id: str, *, map_type: str = "occupancy") -> Path:
    path = Path(settings.MAPS_DIR) / "realtime" / session_id / map_type
    ensure_dir(str(path))
    return path


def default_tile_key(*, t_sec: float, frame_id: str | None = None) -> str:
    if frame_id:
        return f"frame-{frame_id}"
    return f"frame-{int(round(t_sec * 1000.0))}"


def write_stub_map_tile(
    *,
    session_id: str,
    t_sec: float,
    frame_id: str | None,
    map_type: str = "occupancy",
    resolution_m: float | None = None,
    pose: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tile_key = default_tile_key(t_sec=t_sec, frame_id=frame_id)
    out_dir = realtime_tile_dir(session_id, map_type=map_type)
    out_path = out_dir / f"{tile_key}.json"
    payload = {
        "session_id": session_id,
        "frame_id": frame_id,
        "t_sec": float(t_sec),
        "map_type": map_type,
        "resolution_m": float(resolution_m or settings.REALTIME_OCCUPANCY_RESOLUTION_M),
        "pose": pose or {},
        "stats": stats or {},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return normalize_map_tile(
        {
            "tile_key": tile_key,
            "map_type": map_type,
            "storage_path": str(out_path),
            "resolution_m": payload["resolution_m"],
            "bounds": {},
            "stats": payload["stats"],
            "global_map_id": (pose or {}).get("global_map_id"),
        }
    )


class RealtimeOccupancyClient(RealtimeSidecarClient):
    """HTTP client for external realtime occupancy backends."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_sec: float = 10.0,
    ) -> None:
        adapter = create_occupancy_adapter(settings.REALTIME_OCCUPANCY_BACKEND)
        resolved_url = base_url or settings.REALTIME_OCCUPANCY_API_URL or adapter.api_url
        super().__init__(
            backend_name=adapter.name,
            base_url=resolved_url,
            timeout_sec=timeout_sec,
        )

    async def integrate_frame(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.is_configured:
            return None
        data = await self.request_json("POST", "/integrate_frame", payload=payload)
        if not isinstance(data, dict):
            return None
        tile = self.unwrap_dict_payload(data, field="tile")
        if not isinstance(tile, dict):
            return None
        return normalize_map_tile(tile)

    async def fetch_map_tile(self, tile_key: str) -> dict[str, Any] | None:
        if not self.is_configured:
            return None
        data = await self.request_json("GET", f"/map_tile/{tile_key}", allow_404=True)
        if not isinstance(data, dict):
            return None
        tile = self.unwrap_dict_payload(data, field="tile")
        if not isinstance(tile, dict):
            return None
        return normalize_map_tile(tile)
