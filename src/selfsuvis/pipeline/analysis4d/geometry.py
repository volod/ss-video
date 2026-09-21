"""Deterministic geometry used by the 4D contract validator and benchmark.

Predicates are recomputed from stored boxes. Tolerances are explicit. This
module does not run a depth or detection model.
"""

import math

_AXIS_TOL_M = 1e-6
_CONTACT_GAP_M = 0.05
_MOTION_EPS_M = 0.05
_DEFAULT_MAX_SPEED_M_S = 30.0


def aabb(center_m: list[float], extent_m: list[float]) -> tuple[list[float], list[float]]:
    """Return ``(min_xyz, max_xyz)`` for an axis-aligned box."""
    half = [value / 2.0 for value in extent_m]
    low = [center - step for center, step in zip(center_m, half, strict=True)]
    high = [center + step for center, step in zip(center_m, half, strict=True)]
    return low, high


def _overlap(
    a_low: list[float], a_high: list[float], b_low: list[float], b_high: list[float]
) -> bool:
    return all(a_low[i] <= b_high[i] and b_low[i] <= a_high[i] for i in range(3))


def intersects(
    center_a: list[float], extent_a: list[float], center_b: list[float], extent_b: list[float]
) -> bool:
    """True when the two axis-aligned boxes overlap, including boundary contact."""
    a_low, a_high = aabb(center_a, extent_a)
    b_low, b_high = aabb(center_b, extent_b)
    return _overlap(a_low, a_high, b_low, b_high)


def contains(
    outer_center: list[float],
    outer_extent: list[float],
    inner_center: list[float],
    inner_extent: list[float],
) -> bool:
    """True when the inner box lies inside the outer box."""
    outer_low, outer_high = aabb(outer_center, outer_extent)
    inner_low, inner_high = aabb(inner_center, inner_extent)
    return all(outer_low[i] <= inner_low[i] and inner_high[i] <= outer_high[i] for i in range(3))


def box_iou_3d(
    center_a: list[float], extent_a: list[float], center_b: list[float], extent_b: list[float]
) -> float:
    """Intersection over union of two axis-aligned boxes."""
    a_low, a_high = aabb(center_a, extent_a)
    b_low, b_high = aabb(center_b, extent_b)
    inter = 1.0
    for i in range(3):
        inter *= max(0.0, min(a_high[i], b_high[i]) - max(a_low[i], b_low[i]))
    vol_a = extent_a[0] * extent_a[1] * extent_a[2]
    vol_b = extent_b[0] * extent_b[1] * extent_b[2]
    union = vol_a + vol_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_error_m(center_a: list[float], center_b: list[float]) -> float:
    """Euclidean distance between box centers."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(center_a, center_b, strict=True)))


def box_iou_2d(xywh_a: list[float], xywh_b: list[float]) -> float:
    """Intersection over union of two normalized ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = xywh_a
    bx, by, bw, bh = xywh_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _xy_overlap(
    center_a: list[float], extent_a: list[float], center_b: list[float], extent_b: list[float]
) -> bool:
    a_low, a_high = aabb(center_a, extent_a)
    b_low, b_high = aabb(center_b, extent_b)
    return (
        a_low[0] <= b_high[0]
        and b_low[0] <= a_high[0]
        and a_low[1] <= b_high[1]
        and b_low[1] <= a_high[1]
    )


def relation_holds(
    predicate: str,
    subject_center: list[float],
    subject_extent: list[float],
    object_center: list[float],
    object_extent: list[float],
    *,
    distance_band: tuple[float, float] | None = None,
    subject_depth_m: float | None = None,
    object_depth_m: float | None = None,
    subject_motion_m: float | None = None,
) -> bool:
    """Recompute one deterministic predicate. Unknown predicates do not hold."""
    sx, sy, sz = subject_center
    ox, oy, oz = object_center
    if predicate == "contains":
        return contains(subject_center, subject_extent, object_center, object_extent)
    if predicate == "intersects":
        return intersects(subject_center, subject_extent, object_center, object_extent)
    if predicate == "left_of":
        return sx + _AXIS_TOL_M < ox
    if predicate == "right_of":
        return sx > ox + _AXIS_TOL_M
    if predicate == "above":
        return sz > oz + _AXIS_TOL_M
    if predicate == "below":
        return sz + _AXIS_TOL_M < oz
    if predicate == "distance_band":
        if distance_band is None:
            return False
        dist = center_error_m(subject_center, object_center)
        return distance_band[0] <= dist <= distance_band[1]
    if predicate in {"supports", "contacts"}:
        s_low, s_high = aabb(subject_center, subject_extent)
        o_low, o_high = aabb(object_center, object_extent)
        gap = o_low[2] - s_high[2]
        return (
            _xy_overlap(subject_center, subject_extent, object_center, object_extent)
            and abs(gap) <= _CONTACT_GAP_M
        )
    if predicate == "occludes":
        if subject_depth_m is None or object_depth_m is None:
            return False
        return subject_depth_m + _AXIS_TOL_M < object_depth_m and _xy_overlap(
            subject_center, subject_extent, object_center, object_extent
        )
    if predicate == "visible":
        return True
    if predicate == "relative_motion":
        return subject_motion_m is not None and subject_motion_m > _MOTION_EPS_M
    return False


def velocity_feasible(
    samples: list[tuple[float, list[float]]],
    *,
    max_speed_m_s: float = _DEFAULT_MAX_SPEED_M_S,
) -> bool:
    """True when consecutive centers stay within ``max_speed_m_s``.

    Args:
        samples: ``(t_sec, center_m)`` pairs for one subject.
        max_speed_m_s: Speed cap. The default is 30 m/s.

    Returns:
        False when a pair implies a faster translation. A single sample is feasible.
    """
    ordered = sorted(samples, key=lambda item: item[0])
    for (t0, c0), (t1, c1) in zip(ordered, ordered[1:], strict=False):
        dt = t1 - t0
        if dt <= 0:
            return False
        if center_error_m(c0, c1) / dt > max_speed_m_s:
            return False
    return True


def project_pinhole(
    point_m: list[float],
    *,
    focal_px: float,
    principal_px: tuple[float, float],
) -> tuple[float, float]:
    """Project a camera-frame point onto pixels. The camera looks along +Z.

    Args:
        point_m: ``[x, y, z]`` in meters. ``z`` must be positive.
        focal_px: Focal length in pixels.
        principal_px: Principal point ``(u, v)``.

    Returns:
        Pixel ``(u, v)``.
    """
    z = point_m[2]
    if z <= 0 or focal_px <= 0:
        raise ValueError("pinhole projection requires positive depth and focal length")
    u = focal_px * point_m[0] / z + principal_px[0]
    v = focal_px * point_m[1] / z + principal_px[1]
    return u, v


def reprojection_residual_px(
    point_m: list[float],
    observed_px: tuple[float, float],
    *,
    focal_px: float,
    principal_px: tuple[float, float],
) -> float:
    """Euclidean pixel error between a pinhole projection and an observed pixel."""
    u, v = project_pinhole(point_m, focal_px=focal_px, principal_px=principal_px)
    return math.hypot(u - observed_px[0], v - observed_px[1])


def mask_iou(bits_a: str, bits_b: str) -> float:
    """Jaccard index of two equal-length binary masks."""
    if len(bits_a) != len(bits_b) or not bits_a:
        return 0.0
    inter = sum(1 for a, b in zip(bits_a, bits_b, strict=True) if a == "1" and b == "1")
    union = sum(1 for a, b in zip(bits_a, bits_b, strict=True) if a == "1" or b == "1")
    if union == 0:
        return 1.0
    return inter / union


def mask_boundary_f(bits: str, width: int, height: int, other: str) -> float:
    """Boundary F-measure between two masks. A foreground pixel is boundary if a 4-neighbor is background."""

    def boundary(mask: str) -> set[int]:
        found: set[int] = set()
        for index, bit in enumerate(mask):
            if bit != "1":
                continue
            row, col = divmod(index, width)
            neighbors = []
            if col > 0:
                neighbors.append(index - 1)
            if col + 1 < width:
                neighbors.append(index + 1)
            if row > 0:
                neighbors.append(index - width)
            if row + 1 < height:
                neighbors.append(index + width)
            if any(mask[n] == "0" for n in neighbors) or not neighbors:
                found.add(index)
        return found

    if len(bits) != width * height or len(other) != width * height:
        return 0.0
    left, right = boundary(bits), boundary(other)
    if not left and not right:
        return 1.0
    inter = len(left & right)
    if inter == 0:
        return 0.0
    precision = inter / len(left)
    recall = inter / len(right)
    return 2 * precision * recall / (precision + recall)
