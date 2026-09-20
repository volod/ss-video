"""ICP mapper splat fusion lives in ss-video (`pipeline.mapping.mapper`)."""

import os
import shutil
import tempfile
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[3] / "assets" / "splats"


def _identity_4x4():
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class TestFuseSplatFiles:
    def test_produces_fused_ply_on_success(self):
        from selfsuvis.pipeline.mapping.mapper import _fuse_splat_files
        from selfsuvis.pipeline.mapping.splat_io import is_splat_ply, splat_count

        a = str(_ASSETS / "scene_a.ply")
        b = str(_ASSETS / "scene_b.ply")
        with tempfile.TemporaryDirectory() as tmpdir:
            src = shutil.copy(b, os.path.join(tmpdir, "splat.ply"))
            fused_path = _fuse_splat_files(src, a, _identity_4x4(), "m1", "main")
            assert fused_path is not None
            assert os.path.isfile(fused_path)
            assert is_splat_ply(fused_path)
            assert splat_count(fused_path) == splat_count(a) + splat_count(b)

    def test_returns_none_on_bad_source(self):
        from selfsuvis.pipeline.mapping.mapper import _fuse_splat_files

        a = str(_ASSETS / "scene_a.ply")
        result = _fuse_splat_files("/nonexistent.ply", a, _identity_4x4(), "m1", "main")
        assert result is None

    def test_temp_file_cleaned_up(self):
        from selfsuvis.pipeline.mapping.mapper import _fuse_splat_files

        a = str(_ASSETS / "scene_a.ply")
        b = str(_ASSETS / "scene_b.ply")
        with tempfile.TemporaryDirectory() as tmpdir:
            src = shutil.copy(b, os.path.join(tmpdir, "splat.ply"))
            _fuse_splat_files(src, a, _identity_4x4(), "m1", "main")
            aligned_tmp = os.path.join(tmpdir, "_aligned_tmp.ply")
            assert not os.path.exists(aligned_tmp)
