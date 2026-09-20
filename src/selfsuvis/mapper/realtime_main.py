"""Realtime mapper sidecar service.

Provides lightweight project-owned implementations of the realtime mapping API
contract so deployments can start with a consistent HTTP surface before swapping
in heavier SLAM / occupancy engines.
"""

from collections import defaultdict
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from selfsuvis.pipeline.core.env import env_int, env_str, load_layered_env
from selfsuvis.pipeline.realtime import (
    build_fused_pose_from_packets,
    normalize_map_tile,
    normalize_packets,
    normalize_pose_payload,
    write_stub_map_tile,
)

load_layered_env(anchor_file=__file__)

app = FastAPI(title="selfsuvis-realtime-reference", version="1.0.0")
_STATS: dict[str, Any] = {
    "estimate_pose_calls": 0,
    "integrate_frame_calls": 0,
    "tiles_written": 0,
}
_TILE_INDEX: dict[str, dict[str, Any]] = {}
_SESSION_COUNTS: dict[str, int] = defaultdict(int)
_ENGINE_NAME = env_str("REALTIME_ENGINE_NAME", "stub").strip().lower()


class EstimatePoseRequest(BaseModel):
    session_id: str
    packets: list[dict[str, Any]]


class EstimatePoseResponse(BaseModel):
    pose: dict[str, Any] | None


class IntegrateFrameRequest(BaseModel):
    session_id: str
    frame_id: str | None = None
    t_sec: float
    image_path: str = Field(min_length=1)
    depth_path: str | None = None
    map_type: str = "occupancy"
    tile_key: str | None = None
    resolution_m: float = 0.2
    pose: dict[str, Any] | None = None
    packets: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class IntegrateFrameResponse(BaseModel):
    tile: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "realtime-reference", "engine": _ENGINE_NAME}


@app.post("/estimate_pose", response_model=EstimatePoseResponse)
def estimate_pose(body: EstimatePoseRequest) -> EstimatePoseResponse:
    _STATS["estimate_pose_calls"] += 1
    packets = normalize_packets(body.packets)
    pose = build_fused_pose_from_packets(packets, max_lag_ms=120)
    if pose is not None:
        pose["source"] = _ENGINE_NAME
    return EstimatePoseResponse(pose=pose)


@app.post("/integrate_frame", response_model=IntegrateFrameResponse)
def integrate_frame(body: IntegrateFrameRequest) -> IntegrateFrameResponse:
    _STATS["integrate_frame_calls"] += 1
    pose = normalize_pose_payload(body.pose) if body.pose is not None else None
    if pose is None and body.packets:
        pose = build_fused_pose_from_packets(normalize_packets(body.packets), max_lag_ms=120)
    tile = write_stub_map_tile(
        session_id=body.session_id,
        t_sec=body.t_sec,
        frame_id=body.frame_id,
        map_type=body.map_type,
        resolution_m=body.resolution_m,
        pose=pose,
        stats=body.stats,
    )
    if body.tile_key:
        tile["tile_key"] = body.tile_key
    tile = normalize_map_tile(tile)
    _TILE_INDEX[tile["tile_key"]] = tile
    _STATS["tiles_written"] += 1
    _SESSION_COUNTS[body.session_id] += 1
    return IntegrateFrameResponse(tile=tile)


@app.get("/map_tile/{tile_key}")
def get_map_tile(tile_key: str) -> dict[str, Any]:
    tile = _TILE_INDEX.get(tile_key)
    if tile is None:
        raise HTTPException(status_code=404, detail="tile not found")
    return {"tile": tile}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return {
        **_STATS,
        "engine": _ENGINE_NAME,
        "sessions": dict(_SESSION_COUNTS),
        "tile_count": len(_TILE_INDEX),
    }


if __name__ == "__main__":
    port = env_int("PORT", 8101)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
