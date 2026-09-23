"""Back-projection, robust filtering, and gravity-aligned boxes.

Dynamic points are fused in a short motion-compensated window. They are not
inserted into the static cloud. Surface normals come from the depth provider
or from depth gradients, then rotate by the camera pose. Perspective Fields
does not supply those normals.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from selfsuvis.pipeline.analysis4d.camera import CameraModel, project

_UP = np.array([0.0, 0.0, 1.0])
_MIN_POINTS = 12
_MIN_EXTENT_M = 1e-3
_WINDOW_SEC = 1.0


@dataclass(frozen=True)
class OrientedBox:
    """Gravity-aligned box, uncertainty, and the scale label already resolved."""

    center_m: tuple[float, float, float]
    extent_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    covariance_diag: tuple[float, float, float]
    residual_m: float
    observation_count: int
    yaw_rad: float
    metric_scale: str
    dynamic: bool


@dataclass
class StaticCloud:
    """Mission-frame points for tracks that are not dynamic."""

    points: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, track_id: str, samples: np.ndarray, *, dynamic: bool) -> None:
        """Store ``samples`` only when the track is static."""
        if dynamic or samples.size == 0:
            return
        current = self.points.get(track_id)
        stacked = samples if current is None else np.concatenate([current, samples], axis=0)
        self.points[track_id] = stacked


def robust_inliers(points: np.ndarray, k: float = 8.0) -> tuple[np.ndarray, float]:
    """Drop axis-wise outliers. A filled rectangle's corners stay inside the gate.

    The residual is the median absolute deviation of distance from the inlier
    median, in the same units as ``points``.
    """
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(cloud) < _MIN_POINTS:
        return cloud, 0.0
    center = np.median(cloud, axis=0)
    delta = np.abs(cloud - center)
    mad = np.median(delta, axis=0) + 1e-9
    keep = np.all(delta <= k * mad, axis=1)
    if int(keep.sum()) < _MIN_POINTS:
        kept = cloud
    else:
        kept = cloud[keep]
    distance = np.linalg.norm(kept - np.median(kept, axis=0), axis=1)
    residual = float(np.median(np.abs(distance - np.median(distance))))
    return kept, residual


def normals_from_depth(depth: np.ndarray) -> np.ndarray:
    """Camera-frame normals from depth gradients. Invalid depth stays zero."""
    grid = np.asarray(depth, dtype=np.float64)
    safe = np.where(np.isfinite(grid), grid, 0.0)
    dzdy, dzdx = np.gradient(safe)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(safe)], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-9)
    out = normal / norm
    invalid = ~np.isfinite(grid)
    out[invalid] = 0.0
    return out


def transform_normals(normals_cam: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Rotate camera-frame normals into the mission frame. Translation is ignored."""
    rotation = np.asarray(camera.rotation_cam_from_world, dtype=np.float64)
    return np.asarray(normals_cam, dtype=np.float64) @ rotation


def backproject_mask(
    depth: np.ndarray, mask: np.ndarray, camera: CameraModel
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project valid masked pixels.

    Returns:
        World points, pixel centers, and camera-frame depth for those pixels.
    """
    grid = np.asarray(depth, dtype=np.float64)
    keep = np.asarray(mask, dtype=bool)
    if keep.shape != grid.shape:
        keep = _resize_mask(keep, grid.shape[1], grid.shape[0])
    valid = keep & np.isfinite(grid) & (grid > 1e-6)
    if not np.any(valid):
        empty = np.zeros((0, 3))
        return empty, np.zeros((0, 2)), np.zeros((0,))
    vs, us = np.nonzero(valid)
    pixels = np.column_stack([us.astype(np.float64) + 0.5, vs.astype(np.float64) + 0.5])
    z = grid[vs, us]
    return unproject_pixels(pixels, z, camera), pixels, z


def unproject_pixels(pixels: np.ndarray, z_cam: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Back-project using :func:`camera.unproject`."""
    from selfsuvis.pipeline.analysis4d.camera import unproject

    return unproject(pixels, z_cam, camera)


def reprojection_error_px(
    pixels: np.ndarray, points_world: np.ndarray, camera: CameraModel
) -> float:
    """Mean pixel distance after projecting ``points_world``."""
    if len(pixels) == 0:
        return float("inf")
    projected, _z = project(points_world, camera)
    return float(np.linalg.norm(projected - pixels, axis=1).mean())


def fit_gravity_box(
    points: np.ndarray, *, metric_scale: str, dynamic: bool, up: np.ndarray | None = None
) -> OrientedBox | None:
    """Fit a gravity-aligned minimum-volume box to inlier points.

    Yaw is the horizontal minimum-area rectangle. Roll and pitch stay locked
    to ``up`` so the box is not tilted by a partial surface.
    """
    inliers, residual = robust_inliers(points)
    if len(inliers) < _MIN_POINTS:
        return None
    vertical = np.asarray(_UP if up is None else up, dtype=np.float64)
    vertical = vertical / max(float(np.linalg.norm(vertical)), 1e-12)
    horizontal = _horizontal_basis(vertical)
    coords = inliers @ np.column_stack([horizontal[0], horizontal[1], vertical])
    yaw, xr, yr = _min_area_yaw(coords[:, 0], coords[:, 1])
    z = coords[:, 2]
    xmin, xmax = float(xr.min()), float(xr.max())
    ymin, ymax = float(yr.min()), float(yr.max())
    zmin, zmax = float(z.min()), float(z.max())
    center_local = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5])
    c, s = math.cos(yaw), math.sin(yaw)
    center_h = np.array(
        [c * center_local[0] - s * center_local[1], s * center_local[0] + c * center_local[1]]
    )
    center = center_h[0] * horizontal[0] + center_h[1] * horizontal[1] + center_local[2] * vertical
    extent = (
        max(xmax - xmin, _MIN_EXTENT_M),
        max(ymax - ymin, _MIN_EXTENT_M),
        max(zmax - zmin, _MIN_EXTENT_M),
    )
    count = int(len(inliers))
    variance = inliers.var(axis=0) / count
    covariance = (
        float(max(0.0, variance[0])),
        float(max(0.0, variance[1])),
        float(max(0.0, variance[2])),
    )
    forward = math.cos(yaw) * horizontal[0] + math.sin(yaw) * horizontal[1]
    right = np.cross(vertical, forward)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    forward = np.cross(right, vertical)
    rotation = np.column_stack([forward, right, vertical])
    return OrientedBox(
        center_m=_triple(center),
        extent_m=extent,
        quaternion_xyzw=_quaternion_from_matrix(rotation),
        covariance_diag=covariance,
        residual_m=float(max(0.0, residual)),
        observation_count=count,
        yaw_rad=yaw,
        metric_scale=metric_scale,
        dynamic=dynamic,
    )


def compensate_motion(
    stamps: list[tuple[float, np.ndarray]], *, dynamic: bool, window_sec: float = _WINDOW_SEC
) -> np.ndarray:
    """Fuse a short window. Dynamic clouds are translated onto the latest centroid.

    Static clouds are concatenated in the mission frame. The caller decides
    whether the result may enter the static map; this function does not.
    """
    if not stamps:
        return np.zeros((0, 3))
    latest = max(stamp for stamp, _points in stamps)
    recent = [
        (stamp, pts) for stamp, pts in stamps if latest - window_sec <= stamp <= latest + 1e-9
    ]
    if not recent:
        recent = [stamps[-1]]
    if not dynamic or len(recent) == 1:
        return np.concatenate([pts for _stamp, pts in recent], axis=0)
    _t, newest = recent[-1]
    target = newest.mean(axis=0)
    shifted = []
    for _stamp, pts in recent:
        shifted.append(pts + (target - pts.mean(axis=0)))
    return np.concatenate(shifted, axis=0)


def aabb_at_yaw(points: np.ndarray, yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(center, extent)`` of ``points`` in a yaw frame about +Z."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    x = c * points[:, 0] + s * points[:, 1]
    y = -s * points[:, 0] + c * points[:, 1]
    z = points[:, 2]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())
    local = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5])
    center = np.array(
        [
            c * local[0] - s * local[1],
            s * local[0] + c * local[1],
            local[2],
        ]
    )
    extent = np.array(
        [
            max(xmax - xmin, _MIN_EXTENT_M),
            max(ymax - ymin, _MIN_EXTENT_M),
            max(zmax - zmin, _MIN_EXTENT_M),
        ]
    )
    return center, extent


def yaw_error_deg(fitted_rad: float, truth_deg: float) -> float:
    """Smallest edge-angle error. A quarter turn only swaps the horizontal axes."""
    diff = abs(math.degrees(fitted_rad) - truth_deg) % 180.0
    diff = min(diff, 180.0 - diff)
    return float(min(diff, abs(90.0 - diff)))


def _min_area_yaw(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    best_area = float("inf")
    best_yaw = 0.0
    best_xr = x
    best_yr = y
    for step in range(360):
        yaw = math.radians(step * 0.5)
        c, s = math.cos(yaw), math.sin(yaw)
        xr = c * x + s * y
        yr = -s * x + c * y
        area = (float(xr.max()) - float(xr.min())) * (float(yr.max()) - float(yr.min()))
        if area < best_area:
            best_area = area
            best_yaw = yaw
            best_xr = xr
            best_yr = yr
    return best_yaw, best_xr, best_yr


def _horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if abs(float(up[2])) > 0.9:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    tmp = np.array([1.0, 0.0, 0.0]) if abs(float(up[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis_a = np.cross(up, tmp)
    axis_a = axis_a / max(float(np.linalg.norm(axis_a)), 1e-12)
    axis_b = np.cross(up, axis_a)
    axis_b = axis_b / max(float(np.linalg.norm(axis_b)), 1e-12)
    return axis_a, axis_b


def _quaternion_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm == 0:
        return (0.0, 0.0, 0.0, 1.0)
    quat = quat / norm
    if quat[3] < 0:
        quat = -quat
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _triple(values: np.ndarray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.size == 0:
        return np.zeros((height, width), dtype=bool)
    rows = np.linspace(0, mask.shape[0] - 1, height).astype(int)
    cols = np.linspace(0, mask.shape[1] - 1, width).astype(int)
    return mask[rows][:, cols]
