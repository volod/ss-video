"""Synthetic pinhole scene used by the geometry benchmark and unit tests."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from selfsuvis.pipeline.analysis4d.camera import CameraModel, camera_look_at


@dataclass(frozen=True)
class SyntheticScene:
    """One box observed by one calibrated camera."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    eye_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    box_center_m: tuple[float, float, float]
    box_extent_m: tuple[float, float, float]
    box_yaw_deg: float
    calibration_id: str
    reprojection_px_max: float
    box_center_error_m_max: float
    box_extent_error_m_max: float
    yaw_error_deg_max: float
    solid_center_bias_m_max: float


@dataclass(frozen=True)
class SyntheticRender:
    """Depth and the world points that produced it."""

    camera: CameraModel
    depth_m: np.ndarray
    mask: np.ndarray
    pixels: np.ndarray
    points_world: np.ndarray


def load_synthetic_scene(path: Path) -> SyntheticScene:
    """Load the pinned synthetic-camera fixture."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SyntheticScene(
        width=int(payload["width"]),
        height=int(payload["height"]),
        fx=float(payload["fx"]),
        fy=float(payload["fy"]),
        cx=float(payload["cx"]),
        cy=float(payload["cy"]),
        eye_m=tuple(payload["eye_m"]),
        target_m=tuple(payload["target_m"]),
        box_center_m=tuple(payload["box_center_m"]),
        box_extent_m=tuple(payload["box_extent_m"]),
        box_yaw_deg=float(payload["box_yaw_deg"]),
        calibration_id=str(payload["calibration_id"]),
        reprojection_px_max=float(payload["reprojection_px_max"]),
        box_center_error_m_max=float(payload["box_center_error_m_max"]),
        box_extent_error_m_max=float(payload["box_extent_error_m_max"]),
        yaw_error_deg_max=float(payload["yaw_error_deg_max"]),
        solid_center_bias_m_max=float(payload["solid_center_bias_m_max"]),
    )


def render_scene(scene: SyntheticScene) -> SyntheticRender:
    """Ray-cast the box and return camera-frame depth at pixel centers."""
    camera = camera_look_at(
        scene.eye_m,
        scene.target_m,
        width=scene.width,
        height=scene.height,
        fx=scene.fx,
        fy=scene.fy,
        cx=scene.cx,
        cy=scene.cy,
        calibration_id=scene.calibration_id,
        metric_alignment=True,
    )
    us = np.arange(scene.width, dtype=np.float64) + 0.5
    vs = np.arange(scene.height, dtype=np.float64) + 0.5
    grid_u, grid_v = np.meshgrid(us, vs)
    x = (grid_u - camera.cx) / camera.fx
    y = (grid_v - camera.cy) / camera.fy
    dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
    rotation = np.asarray(camera.rotation_cam_from_world, dtype=np.float64)
    dirs_world = np.einsum("ij,hwj->hwi", rotation.T, dirs_cam)
    origin = camera.center_m()
    yaw = math.radians(scene.box_yaw_deg)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    box_rot = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    center = np.asarray(scene.box_center_m, dtype=np.float64)
    half = np.asarray(scene.box_extent_m, dtype=np.float64) * 0.5
    origin_box = box_rot.T @ (origin - center)
    dirs_box = np.einsum("ij,hwj->hwi", box_rot.T, dirs_world)
    t_hit = _ray_aabb(origin_box, dirs_box, -half, half)
    hit = np.isfinite(t_hit)
    points = origin + t_hit[..., None] * dirs_world
    cam = np.einsum("ij,hwj->hwi", rotation, points) + camera.translation_cam_from_world
    depth = np.where(hit, cam[..., 2], np.nan)
    pixels = np.column_stack([grid_u[hit], grid_v[hit]])
    return SyntheticRender(
        camera=camera,
        depth_m=depth,
        mask=hit,
        pixels=pixels,
        points_world=points[hit],
    )


def _ray_aabb(
    origin: np.ndarray, direction: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> np.ndarray:
    """Nearest positive ray hit against an axis-aligned box. Misses are NaN."""
    denom = direction
    near = np.empty_like(direction)
    far = np.empty_like(direction)
    for axis in range(3):
        component = denom[..., axis]
        small = np.abs(component) <= 1e-12
        inside = (origin[axis] >= bmin[axis]) & (origin[axis] <= bmax[axis])
        t1 = np.divide(
            bmin[axis] - origin[axis],
            component,
            out=np.full(component.shape, np.inf),
            where=~small,
        )
        t2 = np.divide(
            bmax[axis] - origin[axis],
            component,
            out=np.full(component.shape, np.inf),
            where=~small,
        )
        t1 = np.where(small & inside, -np.inf, t1)
        t2 = np.where(small & inside, np.inf, t2)
        t1 = np.where(small & ~inside, np.inf, t1)
        t2 = np.where(small & ~inside, -np.inf, t2)
        near[..., axis] = np.minimum(t1, t2)
        far[..., axis] = np.maximum(t1, t2)
    t_near = near.max(axis=-1)
    t_far = far.min(axis=-1)
    t_hit = np.where(t_near > 0, t_near, t_far)
    valid = (t_far >= t_near) & (t_hit > 1e-6) & np.isfinite(t_hit)
    return np.where(valid, t_hit, np.nan)


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    """Rotation about mission +Z."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
