"""Mapper service — thin FastAPI wrapper around pipeline.mapping.icp.

Exposes:
  GET  /health          — liveness probe.
  POST /fuse            — run ICP registration between two splat.ply files.
  POST /check_overlap   — GPS pre-check before attempting ICP.

Called by the worker (pipeline/mapper.py) after splatfacto reconstruction
completes, following the same HTTP pattern as the nerfstudio wrapper.

Environment:
  DATA_DIR  — base data directory (default ./.data).
  PORT      — listen port (default 8000).
"""

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from selfsuvis.pipeline.core.env import env_int, load_layered_env
from selfsuvis.pipeline.mapping.icp import IcpResult, check_overlap, register_splats
from selfsuvis.pipeline.mapping.splat_io import read_splat_metadata

load_layered_env(anchor_file=__file__)

app = FastAPI(title="selfsuvis-mapper", version="1.0.0")


# -- request / response models -------------------------------------------------


class FuseRequest(BaseModel):
    source_path: str = Field(..., description="Path to source splat.ply (new mission)")
    target_path: str = Field(..., description="Path to target splat.ply (reference)")
    source_meta: dict[str, Any] | None = Field(
        default=None,
        description="GPS origin dict {origin_lat, origin_lon, origin_alt}. "
        "Auto-loaded from <source>_meta.json if omitted.",
    )
    target_meta: dict[str, Any] | None = Field(
        default=None,
        description="GPS origin dict for target. Auto-loaded if omitted.",
    )
    max_correspondence_m: float = Field(default=2.0, gt=0)
    max_iterations: int = Field(default=100, ge=1, le=1000)
    voxel_size_m: float = Field(default=0.0, ge=0.0, description="0 = auto")


class FuseResponse(BaseModel):
    status: str  # "ok" | "no_overlap" | "error"
    transform_4x4: list[list[float]] | None
    rmse: float | None
    fitness: float | None
    converged: bool
    message: str


class OverlapRequest(BaseModel):
    source_meta: dict[str, Any]
    target_meta: dict[str, Any]
    radius_a_m: float = Field(default=15.0, gt=0)
    radius_b_m: float = Field(default=15.0, gt=0)


class OverlapResponse(BaseModel):
    overlaps: bool
    gps_distance_m: float


# -- endpoints -----------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mapper"}


@app.post("/check_overlap", response_model=OverlapResponse)
def api_check_overlap(req: OverlapRequest) -> OverlapResponse:
    """GPS-based overlap pre-check. Call before /fuse to skip obviously disjoint scenes."""
    overlaps, dist = check_overlap(
        req.source_meta,
        req.target_meta,
        req.radius_a_m,
        req.radius_b_m,
    )
    return OverlapResponse(overlaps=overlaps, gps_distance_m=round(dist, 2))


@app.post("/fuse", response_model=FuseResponse)
def api_fuse(req: FuseRequest) -> FuseResponse:
    """Run ICP registration of source into target coordinate frame.

    Returns the SE(3) transform (source → target), ICP residual RMSE,
    and fitness score. The caller (worker) persists the result in
    global_map_missions.registration_transform_json.
    """
    # Auto-load metadata from companion JSON if not provided
    source_meta = req.source_meta or read_splat_metadata(req.source_path)
    target_meta = req.target_meta or read_splat_metadata(req.target_path)

    # GPS pre-check (skip ICP if scenes clearly don't overlap)
    if source_meta and target_meta:
        overlaps, dist = check_overlap(source_meta, target_meta)
        if not overlaps:
            return FuseResponse(
                status="no_overlap",
                transform_4x4=None,
                rmse=None,
                fitness=0.0,
                converged=False,
                message=f"GPS distance {dist:.1f}m exceeds combined scene radii — skipping ICP",
            )

    try:
        result: IcpResult = register_splats(
            source_path=req.source_path,
            target_path=req.target_path,
            source_meta=source_meta,
            target_meta=target_meta,
            max_correspondence_m=req.max_correspondence_m,
            max_iterations=req.max_iterations,
            voxel_size_m=req.voxel_size_m,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"open3d not available: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ICP failed: {exc}")

    return FuseResponse(
        status="ok",
        transform_4x4=result.transform_4x4,
        rmse=round(result.rmse, 6),
        fitness=round(result.fitness, 6),
        converged=result.converged,
        message=result.message,
    )


if __name__ == "__main__":
    port = env_int("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
