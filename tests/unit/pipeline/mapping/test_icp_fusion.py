"""Unit tests for pipeline.icp_fusion.

Tests that don't require open3d use the GPS/overlap helpers directly.
Tests that call register_splats are skipped when open3d is not installed.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

_ASSETS = Path(__file__).resolve().parents[3] / "assets" / "splats"

# ── open3d availability ───────────────────────────────────────────────────────

try:
    import open3d  # noqa: F401

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False

skip_no_open3d = pytest.mark.skipif(not _HAS_OPEN3D, reason="open3d not installed")


# ── fixtures ──────────────────────────────────────────────────────────────────

_META_A: dict[str, Any] = {"origin_lat": 48.0, "origin_lon": 11.0, "origin_alt": 100.0}
_META_B: dict[str, Any] = {
    "origin_lat": 48.00005,
    "origin_lon": 11.00005,
    "origin_alt": 100.0,
}  # ~7m NE
_META_C: dict[str, Any] = {
    "origin_lat": 48.05,
    "origin_lon": 11.05,
    "origin_alt": 100.0,
}  # ~6km away


# ── check_overlap ─────────────────────────────────────────────────────────────


def test_overlap_nearby_scenes():
    from selfsuvis.pipeline.mapping.icp import check_overlap

    overlaps, dist = check_overlap(_META_A, _META_B, radius_a_m=10.0, radius_b_m=10.0)
    assert overlaps is True
    assert dist < 20.0  # ~7m apart, well within combined 20m radius


def test_overlap_distant_scenes():
    from selfsuvis.pipeline.mapping.icp import check_overlap

    overlaps, dist = check_overlap(_META_A, _META_C, radius_a_m=10.0, radius_b_m=10.0)
    assert overlaps is False
    assert dist > 5_000.0


def test_overlap_same_scene():
    from selfsuvis.pipeline.mapping.icp import check_overlap

    overlaps, dist = check_overlap(_META_A, _META_A)
    assert overlaps is True
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_overlap_distance_is_symmetric():
    from selfsuvis.pipeline.mapping.icp import check_overlap

    _, d1 = check_overlap(_META_A, _META_B)
    _, d2 = check_overlap(_META_B, _META_A)
    assert abs(d1 - d2) < 0.01


def test_overlap_distance_matches_gps_calculation():
    """GPS distance between scene_a and scene_b should be ~7m."""
    from selfsuvis.pipeline.mapping.icp import check_overlap

    _, dist = check_overlap(_META_A, _META_B)
    # 48.00005 - 48.0 = 5e-5 deg lat = 5.57m; lon similarly ~3.7m → total ~6.7m
    assert 5.0 < dist < 10.0


# ── _initial_transform_from_gps ──────────────────────────────────────────────


def test_initial_transform_identity_same_origin():
    from selfsuvis.pipeline.mapping.icp import _initial_transform_from_gps

    T = _initial_transform_from_gps(_META_A, _META_A)
    np.testing.assert_allclose(T[:3, 3], [0, 0, 0], atol=1e-3)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-6)


def test_initial_transform_nonzero_for_offset():
    from selfsuvis.pipeline.mapping.icp import _initial_transform_from_gps

    T = _initial_transform_from_gps(_META_B, _META_A)
    # source_b is ~7m NE of target_a
    translation = T[:3, 3]
    dist = np.linalg.norm(translation)
    assert 5.0 < dist < 10.0, f"expected ~7m translation, got {dist:.2f}m"


def test_initial_transform_is_4x4():
    from selfsuvis.pipeline.mapping.icp import _initial_transform_from_gps

    T = _initial_transform_from_gps(_META_A, _META_B)
    assert T.shape == (4, 4)
    assert T[3, 3] == pytest.approx(1.0)


def test_initial_transform_rotation_is_identity():
    """Phase 1 = pure translation; rotation block must be identity."""
    from selfsuvis.pipeline.mapping.icp import _initial_transform_from_gps

    T = _initial_transform_from_gps(_META_A, _META_B)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-6)


# ── _voxel_size_for ───────────────────────────────────────────────────────────


def test_voxel_size_zero_for_small_clouds():
    from selfsuvis.pipeline.mapping.icp import _voxel_size_for

    assert _voxel_size_for(100) == 0.0
    assert _voxel_size_for(5_000) == 0.0


def test_voxel_size_positive_for_large_clouds():
    from selfsuvis.pipeline.mapping.icp import _voxel_size_for

    v = _voxel_size_for(100_000)
    assert v > 0.0


def test_voxel_size_minimum_is_5cm():
    from selfsuvis.pipeline.mapping.icp import _voxel_size_for

    v = _voxel_size_for(10_000)
    assert v >= 0.05


# ── register_splats (requires open3d) ─────────────────────────────────────────


@skip_no_open3d
def test_register_overlapping_scenes_converges():
    """scene_a and scene_b overlap (~7m offset) — ICP should converge."""
    from selfsuvis.pipeline.mapping.icp import register_splats
    from selfsuvis.pipeline.mapping.splat_io import read_splat_metadata

    src = str(_ASSETS / "scene_b.ply")
    tgt = str(_ASSETS / "scene_a.ply")
    src_meta = read_splat_metadata(src)
    tgt_meta = read_splat_metadata(tgt)

    result = register_splats(
        source_path=src,
        target_path=tgt,
        source_meta=src_meta,
        target_meta=tgt_meta,
        max_correspondence_m=5.0,
        max_iterations=50,
    )
    assert result.converged
    assert result.fitness > 0.0
    assert result.rmse < 5.0
    assert len(result.transform_4x4) == 4
    assert len(result.transform_4x4[0]) == 4


@skip_no_open3d
def test_register_returns_valid_se3():
    """Transform matrix should have det ≈ 1 (rotation part is proper rotation)."""
    from selfsuvis.pipeline.mapping.icp import register_splats
    from selfsuvis.pipeline.mapping.splat_io import read_splat_metadata

    result = register_splats(
        source_path=str(_ASSETS / "scene_b.ply"),
        target_path=str(_ASSETS / "scene_a.ply"),
        source_meta=read_splat_metadata(str(_ASSETS / "scene_b.ply")),
        target_meta=read_splat_metadata(str(_ASSETS / "scene_a.ply")),
        max_correspondence_m=5.0,
    )
    T = np.array(result.transform_4x4)
    det = np.linalg.det(T[:3, :3])
    assert abs(det - 1.0) < 0.01, f"rotation det={det:.4f}, expected ~1.0"
    assert T[3, 3] == pytest.approx(1.0, abs=1e-5)


@skip_no_open3d
def test_register_nonoverlapping_low_fitness():
    """scene_a and scene_c don't overlap — fitness should be low."""
    from selfsuvis.pipeline.mapping.icp import register_splats
    from selfsuvis.pipeline.mapping.splat_io import read_splat_metadata

    result = register_splats(
        source_path=str(_ASSETS / "scene_c.ply"),
        target_path=str(_ASSETS / "scene_a.ply"),
        source_meta=read_splat_metadata(str(_ASSETS / "scene_c.ply")),
        target_meta=read_splat_metadata(str(_ASSETS / "scene_a.ply")),
        max_correspondence_m=5.0,
    )
    # Non-overlapping: fitness near 0, not converged
    assert result.fitness < 0.3


@skip_no_open3d
def test_register_missing_source_raises():
    from selfsuvis.pipeline.mapping.icp import register_splats

    with pytest.raises(FileNotFoundError):
        register_splats("/nonexistent/source.ply", str(_ASSETS / "scene_a.ply"))


@skip_no_open3d
def test_register_no_meta_uses_identity_init():
    """Without metadata, should still run (identity initial alignment)."""
    from selfsuvis.pipeline.mapping.icp import register_splats

    result = register_splats(
        source_path=str(_ASSETS / "scene_b.ply"),
        target_path=str(_ASSETS / "scene_a.ply"),
        source_meta=None,
        target_meta=None,
        max_correspondence_m=5.0,
    )
    # May or may not converge without GPS init, but should return a valid result
    assert result.transform_4x4 is not None
    assert isinstance(result.fitness, float)


# ── IcpResult dataclass ───────────────────────────────────────────────────────


def test_icp_result_fields():
    from selfsuvis.pipeline.mapping.icp import IcpResult

    r = IcpResult(
        transform_4x4=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        rmse=0.05,
        fitness=0.8,
        converged=True,
        n_source=200,
        n_target=200,
        voxel_size_m=0.0,
    )
    assert r.converged is True
    assert r.rmse == pytest.approx(0.05)
    assert r.message == ""
