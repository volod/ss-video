"""Pinhole cameras in the ss-mapping pose convention.

``pipeline.mapping.sfm`` stores ``cam_from_world`` as ``R`` and ``t``:

    X_cam = R @ X_world + t

A pose dict with those keys back-projects in the same frame. Pixel ``z`` is
the camera-frame depth, not the euclidean range. Gravity for a mission ENU
frame is +Z.
"""

import math
from dataclasses import dataclass

import numpy as np

_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class CameraModel:
    """Intrinsics, radial distortion, and a cam-from-world pose."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation_cam_from_world: np.ndarray
    translation_cam_from_world: np.ndarray
    k1: float = 0.0
    k2: float = 0.0
    calibration_id: str | None = None
    metric_alignment: bool = False
    pose_known: bool = True
    roll_deg: float | None = None
    pitch_deg: float | None = None
    fov_deg: float | None = None

    @property
    def intrinsics_ok(self) -> bool:
        return (
            self.width > 1
            and self.height > 1
            and self.fx > 0
            and self.fy > 0
            and math.isfinite(self.fx)
            and math.isfinite(self.fy)
        )

    @property
    def pose_ok(self) -> bool:
        rotation = np.asarray(self.rotation_cam_from_world, dtype=np.float64)
        translation = np.asarray(self.translation_cam_from_world, dtype=np.float64)
        return (
            rotation.shape == (3, 3)
            and translation.shape == (3,)
            and bool(np.isfinite(rotation).all() and np.isfinite(translation).all())
        )

    def center_m(self) -> np.ndarray:
        """Camera origin in the mission frame."""
        rotation = np.asarray(self.rotation_cam_from_world, dtype=np.float64)
        translation = np.asarray(self.translation_cam_from_world, dtype=np.float64)
        return -rotation.T @ translation


@dataclass(frozen=True)
class PerspectiveEstimate:
    """Roll, pitch, and field of view. This is not a metric scale and not a normal map."""

    roll_deg: float
    pitch_deg: float
    fov_deg: float
    confidence: float


def camera_from_mapping_pose(
    pose: dict,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    calibration_id: str | None = None,
    metric_alignment: bool = False,
    k1: float = 0.0,
    k2: float = 0.0,
) -> CameraModel:
    """Build a camera from an ss-mapping ``{"R", "t"}`` pose dict.

    Args:
        pose: ``R`` is cam-from-world (3x3) and ``t`` is the matching translation.
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length in pixels.
        fy: Focal length in pixels.
        cx: Principal point.
        cy: Principal point.
        calibration_id: Set only when the pose and intrinsics are a real calibration.
        metric_alignment: True when scale comes from SLAM, GPS/IMU, or a known height.
        k1: Radial distortion.
        k2: Radial distortion.

    Returns:
        A camera that uses the mapping transform unchanged.
    """
    rotation = np.asarray(pose["R"], dtype=np.float64)
    translation = np.asarray(pose["t"], dtype=np.float64).reshape(3)
    return CameraModel(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rotation_cam_from_world=rotation,
        translation_cam_from_world=translation,
        k1=k1,
        k2=k2,
        calibration_id=calibration_id,
        metric_alignment=metric_alignment,
    )


def camera_look_at(
    eye_m: tuple[float, float, float],
    target_m: tuple[float, float, float],
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    calibration_id: str | None = None,
    metric_alignment: bool = False,
    k1: float = 0.0,
    k2: float = 0.0,
) -> CameraModel:
    """Camera at ``eye_m`` looking at ``target_m``, with mission +Z as up.

    The optical axis is camera +Z. Camera +X points right and camera +Y points
    down, then both are converted to the mapping ``R``, ``t`` pair.
    """
    eye = np.asarray(eye_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    forward = _unit(target - eye)
    right = _unit(np.cross(forward, _UP))
    down = _unit(np.cross(forward, right))
    world_from_cam = np.column_stack([right, down, forward])
    rotation = world_from_cam.T
    translation = -rotation @ eye
    return CameraModel(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rotation_cam_from_world=rotation,
        translation_cam_from_world=translation,
        k1=k1,
        k2=k2,
        calibration_id=calibration_id,
        metric_alignment=metric_alignment,
    )


def mapping_pose_dict(camera: CameraModel) -> dict[str, list]:
    """Return the ``R``/``t`` dict ss-mapping SfM writes for this camera."""
    return {
        "R": np.asarray(camera.rotation_cam_from_world, dtype=np.float64).tolist(),
        "t": np.asarray(camera.translation_cam_from_world, dtype=np.float64).reshape(3).tolist(),
    }


def apply_perspective_fields(
    estimate: PerspectiveEstimate,
    *,
    width: int,
    height: int,
    camera: CameraModel | None = None,
) -> CameraModel:
    """Fill missing intrinsics from field of view. Scale stays non-metric.

    Roll and pitch are stored for gravity regularization. They do not become
    surface normals, and ``metric_alignment`` is forced off when the previous
    camera had no calibration id.
    """
    fov = max(1e-3, min(179.0, float(estimate.fov_deg)))
    fy = (height * 0.5) / math.tan(math.radians(fov) * 0.5)
    fx = fy
    pose_known = False
    rotation = np.eye(3)
    translation = np.zeros(3)
    k1 = k2 = 0.0
    calibration_id = None
    metric_alignment = False
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    if camera is not None:
        if camera.intrinsics_ok:
            fx, fy = camera.fx, camera.fy
            cx, cy = camera.cx, camera.cy
        if camera.pose_known and camera.pose_ok:
            rotation = np.asarray(camera.rotation_cam_from_world, dtype=np.float64)
            translation = np.asarray(camera.translation_cam_from_world, dtype=np.float64).reshape(3)
            pose_known = True
        k1, k2 = camera.k1, camera.k2
        calibration_id = camera.calibration_id
        metric_alignment = bool(camera.metric_alignment and camera.calibration_id and pose_known)
    return CameraModel(
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        rotation_cam_from_world=np.asarray(rotation, dtype=np.float64),
        translation_cam_from_world=np.asarray(translation, dtype=np.float64).reshape(3),
        k1=k1,
        k2=k2,
        calibration_id=calibration_id,
        metric_alignment=metric_alignment,
        pose_known=pose_known,
        roll_deg=float(estimate.roll_deg),
        pitch_deg=float(estimate.pitch_deg),
        fov_deg=fov,
    )


def project(points_world: np.ndarray, camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixels and camera-frame depth.

    Returns:
        ``(pixels, z_cam)`` where ``pixels`` has shape ``(N, 2)``.
    """
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(camera.rotation_cam_from_world, dtype=np.float64)
    translation = np.asarray(camera.translation_cam_from_world, dtype=np.float64).reshape(3)
    cam = points @ rotation.T + translation
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-12, np.nan, z)
    x = cam[:, 0] / safe
    y = cam[:, 1] / safe
    x, y = _distort(x, y, camera.k1, camera.k2)
    pixels = np.column_stack([camera.fx * x + camera.cx, camera.fy * y + camera.cy])
    return pixels, z


def unproject(pixels: np.ndarray, z_cam: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Back-project pixels at camera-frame depth into the mission frame."""
    uv = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(z_cam, dtype=np.float64).reshape(-1)
    x = (uv[:, 0] - camera.cx) / camera.fx
    y = (uv[:, 1] - camera.cy) / camera.fy
    x, y = _undistort(x, y, camera.k1, camera.k2)
    cam = np.column_stack([x * depth, y * depth, depth])
    rotation = np.asarray(camera.rotation_cam_from_world, dtype=np.float64)
    translation = np.asarray(camera.translation_cam_from_world, dtype=np.float64).reshape(3)
    return (cam - translation) @ rotation


def resolve_scale(
    provider_scale: str,
    camera: CameraModel | None,
) -> str:
    """Return ``metric``, ``relative``, or ``unavailable``.

    Relative depth stays relative even when a metric pose is present. Metric
    depth without a calibration id or a metric alignment is relative. A missing
    pose or missing intrinsics is unavailable. Nothing in this function invents
    meters from a relative map.
    """
    if provider_scale not in {"metric", "relative"}:
        return "unavailable"
    if camera is None or not camera.intrinsics_ok or not camera.pose_ok or not camera.pose_known:
        return "unavailable"
    if provider_scale != "metric":
        return "relative"
    if not camera.calibration_id or not camera.metric_alignment:
        return "relative"
    return "metric"


def _distort(x: np.ndarray, y: np.ndarray, k1: float, k2: float) -> tuple[np.ndarray, np.ndarray]:
    if k1 == 0.0 and k2 == 0.0:
        return x, y
    r2 = x * x + y * y
    scale = 1.0 + k1 * r2 + k2 * r2 * r2
    return x * scale, y * scale


def _undistort(
    x: np.ndarray, y: np.ndarray, k1: float, k2: float, steps: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    if k1 == 0.0 and k2 == 0.0:
        return x, y
    xu, yu = x.copy(), y.copy()
    for _ in range(steps):
        r2 = xu * xu + yu * yu
        scale = 1.0 + k1 * r2 + k2 * r2 * r2
        scale = np.where(np.abs(scale) < 1e-8, np.nan, scale)
        xu, yu = x / scale, y / scale
    return xu, yu


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("camera axis length is zero")
    return vector / norm
