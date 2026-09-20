"""Phase 2 global map: 3DGS ICP fusion via Open3D.

Registers per-mission splat.ply files into a shared ENU coordinate frame using
Point-to-Point ICP (Iterative Closest Point). The input to ICP is the Gaussian
centre positions extracted from each splat — a (N, 3) point cloud in ENU metres.

Phase 1 GPS-to-ENU registration provides the initial alignment (coarse); ICP
refines it to sub-metre accuracy when the scenes overlap sufficiently.

Typical usage (called by mapper/main.py):
    result = register_splats(
        source_path=".data/maps/mission_b/splat.ply",
        target_path=".data/maps/mission_a/splat.ply",
        source_meta={"origin_lat": ..., "origin_lon": ..., "origin_alt": ...},
        target_meta={"origin_lat": ..., "origin_lon": ..., "origin_alt": ...},
        max_correspondence_m=2.0,
    )
    # result.transform_4x4   — SE(3) 4×4 list[[float]]  (source → target frame)
    # result.rmse            — ICP residual in metres
    # result.fitness         — overlap fraction [0, 1]
    # result.converged       — True if ICP converged within criteria
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IcpResult:
    """Result of an ICP registration attempt."""

    transform_4x4: list[list[float]]  # SE(3) 4×4 matrix (source → target)
    rmse: float  # correspondence RMSE in metres
    fitness: float  # overlap fraction [0, 1]
    converged: bool
    n_source: int  # input point count
    n_target: int
    voxel_size_m: float  # downsampling used (0 = none)
    message: str = ""


# -- helpers -------------------------------------------------------------------


def _initial_transform_from_gps(
    source_meta: dict[str, Any],
    target_meta: dict[str, Any],
) -> np.ndarray:
    """Compute the Phase-1 GPS translation as a 4×4 SE(3) matrix.

    Both metas must have origin_lat, origin_lon, origin_alt.
    Returns the transform that moves the source ENU origin to the target ENU frame.
    """
    from selfsuvis.pipeline.mapping.gps_registration import gps_to_enu

    # Express source origin in target's ENU frame
    e, n, u = gps_to_enu(
        lat=source_meta["origin_lat"],
        lon=source_meta["origin_lon"],
        alt=source_meta["origin_alt"],
        origin_lat=target_meta["origin_lat"],
        origin_lon=target_meta["origin_lon"],
        origin_alt=target_meta["origin_alt"],
    )
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = e
    T[1, 3] = n
    T[2, 3] = u
    return T


def _to_open3d_pointcloud(positions: np.ndarray):
    """Convert (N, 3) float32 array to an open3d PointCloud."""
    import open3d as o3d  # type: ignore

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    return pcd


def _voxel_size_for(n_points: int, target_points: int = 5_000) -> float:
    """Heuristic voxel downsampling size so ICP runs in reasonable time.

    Returns 0.0 (no downsampling) when n_points <= target_points.
    """
    if n_points <= target_points:
        return 0.0
    # Approximate: cube root ratio of point counts
    ratio = (n_points / target_points) ** (1 / 3)
    # Round to sensible grid
    raw = ratio * 0.05  # 5cm base voxel
    return max(0.05, round(raw, 2))


# -- main API ------------------------------------------------------------------


def register_splats(
    source_path: str,
    target_path: str,
    source_meta: dict[str, Any] | None = None,
    target_meta: dict[str, Any] | None = None,
    max_correspondence_m: float = 2.0,
    max_iterations: int = 100,
    voxel_size_m: float = 0.0,
) -> IcpResult:
    """Register source splat.ply into the target splat.ply coordinate frame.

    Args:
        source_path: path to the splat.ply to align (the "new" mission).
        target_path: path to the reference splat.ply (the "existing" scene).
        source_meta: GPS metadata dict with origin_lat/lon/alt.
                     If provided together with target_meta, used as initial alignment.
                     If None, uses identity as initial guess.
        target_meta: GPS metadata dict for the target scene.
        max_correspondence_m: ICP max correspondence distance in metres.
                               Rule of thumb: 2× expected GPS error.
        max_iterations: ICP iteration cap.
        voxel_size_m: Voxel downsampling grid size (0 = auto-select).

    Returns:
        IcpResult with transform, fitness, rmse, and convergence info.

    Raises:
        ImportError: if open3d is not installed.
        FileNotFoundError: if either PLY file does not exist.
    """
    import open3d as o3d  # type: ignore

    from selfsuvis.pipeline.mapping.splat_io import splat_positions

    # Load positions
    src_pts = splat_positions(source_path)
    tgt_pts = splat_positions(target_path)

    n_src, n_tgt = len(src_pts), len(tgt_pts)

    # Auto voxel size
    if voxel_size_m == 0.0:
        voxel_size_m = max(
            _voxel_size_for(n_src),
            _voxel_size_for(n_tgt),
        )

    # Build open3d point clouds
    src_pcd = _to_open3d_pointcloud(src_pts)
    tgt_pcd = _to_open3d_pointcloud(tgt_pts)

    if voxel_size_m > 0.0:
        src_pcd = src_pcd.voxel_down_sample(voxel_size_m)
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size_m)

    # Initial alignment from Phase 1 GPS registration
    if (
        source_meta
        and target_meta
        and all(k in source_meta for k in ("origin_lat", "origin_lon", "origin_alt"))
        and all(k in target_meta for k in ("origin_lat", "origin_lon", "origin_alt"))
    ):
        init_T = _initial_transform_from_gps(source_meta, target_meta)
    else:
        init_T = np.eye(4, dtype=np.float64)

    # ICP registration (Point-to-Point)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
        relative_fitness=1e-6,
        relative_rmse=1e-6,
    )
    result = o3d.pipelines.registration.registration_icp(
        source=src_pcd,
        target=tgt_pcd,
        max_correspondence_distance=max_correspondence_m,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=criteria,
    )

    T = result.transformation.tolist()
    fitness = float(result.fitness)
    rmse = float(result.inlier_rmse)
    converged = fitness > 0.0 and rmse < max_correspondence_m

    msg = (
        f"ICP: fitness={fitness:.3f} rmse={rmse:.4f}m "
        f"src={n_src} tgt={n_tgt} voxel={voxel_size_m:.2f}m"
    )

    return IcpResult(
        transform_4x4=T,
        rmse=rmse,
        fitness=fitness,
        converged=converged,
        n_source=n_src,
        n_target=n_tgt,
        voxel_size_m=voxel_size_m,
        message=msg,
    )


def check_overlap(
    source_meta: dict[str, Any],
    target_meta: dict[str, Any],
    radius_a_m: float = 15.0,
    radius_b_m: float = 15.0,
) -> tuple[bool, float]:
    """Quick GPS check: do two scenes overlap enough to attempt ICP?

    Uses the GPS ENU origins and approximate scene radii to estimate overlap.
    Returns (overlaps: bool, gps_distance_m: float).
    """
    from selfsuvis.pipeline.mapping.gps_registration import gps_to_enu

    e, n, u = gps_to_enu(
        lat=source_meta["origin_lat"],
        lon=source_meta["origin_lon"],
        alt=source_meta["origin_alt"],
        origin_lat=target_meta["origin_lat"],
        origin_lon=target_meta["origin_lon"],
        origin_alt=target_meta["origin_alt"],
    )
    dist = math.sqrt(e * e + n * n + u * u)
    overlaps = dist < (radius_a_m + radius_b_m)
    return overlaps, dist
