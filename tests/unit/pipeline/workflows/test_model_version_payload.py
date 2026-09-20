"""Unit tests for model_version_id in Qdrant frame payload.

Verifies that pipeline.workflows.indexer.VideoIndexer._build_frame_point always
includes 'model_version_id' in the PointStruct payload, sourced from
pipeline.core.config.settings.MODEL_VERSION_ID.

cv2 and skimage are stubbed because they fail to import under NumPy 2.x
in this environment. All other deps (qdrant_client, torch, PIL) are real.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ── Import the module under test ─────────────────────────────────────────────
# cv2 4.13+ works with NumPy 2.x — stubs are not needed. If cv2 is missing
# entirely, temporarily stub it so the indexer import succeeds, then restore
# sys.modules so subsequent tests that need the real cv2 are not affected.
if "selfsuvis.pipeline.workflows.indexer" not in sys.modules:
    _stub_names = ("cv2", "skimage", "skimage.metrics")
    _saved = {n: sys.modules.get(n) for n in _stub_names}
    _cv2_missing = False
    try:
        import cv2 as _cv2_probe  # noqa: F401
    except Exception:
        _cv2_missing = True

    if _cv2_missing:
        for _name in _stub_names:
            _m = types.ModuleType(_name)
            _m.__spec__ = type(
                "S",
                (),
                {
                    "name": _name,
                    "origin": None,
                    "loader": None,
                    "submodule_search_locations": None,
                    "parent": _name.rsplit(".", 1)[0] if "." in _name else "",
                    "has_location": False,
                },
            )()
            sys.modules[_name] = _m

    indexer_mod = importlib.import_module("selfsuvis.pipeline.workflows.indexer")

    if _cv2_missing:
        for _name in _stub_names:
            _orig = _saved[_name]
            if _orig is None:
                sys.modules.pop(_name, None)
            else:
                sys.modules[_name] = _orig

    del _stub_names, _saved, _cv2_missing
else:
    indexer_mod = importlib.import_module("selfsuvis.pipeline.workflows.indexer")

VideoIndexer = indexer_mod.VideoIndexer  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_indexer(model_version_id="base", clip_dim=512):
    """Build a VideoIndexer shell with mocked attributes — no model loading."""
    settings_mock = MagicMock()
    settings_mock.MODEL_VERSION_ID = model_version_id

    indexer = VideoIndexer.__new__(VideoIndexer)
    indexer.clip = MagicMock()
    indexer.dino_model = None  # checked inside _build_frame_point
    indexer.store = MagicMock()
    return indexer, settings_mock


def _build_point(indexer, settings_mock, **kwargs):
    """Call _build_frame_point with patched settings."""
    import numpy as np
    from PIL import Image as PILImage

    fake_pil = PILImage.new("RGB", (32, 32))
    fake_clip = np.zeros(512, dtype=np.float32)
    defaults = dict(
        video_id="vid1",
        segment_id=0,
        t_sec=1.0,
        frame_path="/tmp/frame.jpg",
        frame_pil=fake_pil,
        clip_embed=fake_clip,
    )
    defaults.update(kwargs)
    import selfsuvis.pipeline.workflows.indexer as _wf_indexer

    with patch.object(_wf_indexer._core, "settings", settings_mock):
        return VideoIndexer._build_frame_point(indexer, **defaults)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestModelVersionIdInPayload:
    def test_default_base_version_present(self):
        indexer, settings_mock = _make_indexer(model_version_id="base")
        point = _build_point(indexer, settings_mock)
        assert "model_version_id" in point.payload
        assert point.payload["model_version_id"] == "base"

    def test_custom_version_stored_correctly(self):
        indexer, settings_mock = _make_indexer(model_version_id="sup_abc12345")
        point = _build_point(indexer, settings_mock)
        assert point.payload["model_version_id"] == "sup_abc12345"

    def test_version_reflects_current_settings(self):
        """Two calls with different MODEL_VERSION_ID produce different payloads."""
        idx1, s1 = _make_indexer(model_version_id="v1")
        idx2, s2 = _make_indexer(model_version_id="v2")
        p1 = _build_point(idx1, s1, t_sec=1.0)
        p2 = _build_point(idx2, s2, t_sec=2.0)
        assert p1.payload["model_version_id"] == "v1"
        assert p2.payload["model_version_id"] == "v2"

    def test_model_version_present_with_all_standard_fields(self):
        indexer, settings_mock = _make_indexer()
        point = _build_point(
            indexer,
            settings_mock,
            video_id="vid99",
            segment_id=3,
            t_sec=5.0,
            frame_path="/tmp/frame.jpg",
            mission_id="mission-1",
            robot_id="rover-a",
        )
        p = point.payload
        assert p["video_id"] == "vid99"
        assert p["segment_id"] == 3
        assert p["t_sec"] == 5.0
        assert p["frame_path"] == "/tmp/frame.jpg"
        assert p["mission_id"] == "mission-1"
        assert p["robot_id"] == "rover-a"
        assert p["model_version_id"] == "base"

    def test_model_version_present_without_optional_fields(self):
        """model_version_id always written even when optional fields are absent."""
        indexer, settings_mock = _make_indexer()
        point = _build_point(indexer, settings_mock)
        assert "model_version_id" in point.payload
        # Optional fields absent when not provided
        assert "gps" not in point.payload
        assert "enu" not in point.payload
        assert "global_map_id" not in point.payload

    def test_model_version_present_with_gps(self):
        indexer, settings_mock = _make_indexer(model_version_id="v3")
        point = _build_point(
            indexer,
            settings_mock,
            gps={"lat": 37.7749, "lon": -122.4194, "alt": 10.0},
        )
        assert point.payload["model_version_id"] == "v3"
        assert point.payload["gps"]["lat"] == pytest.approx(37.7749)

    def test_point_id_is_deterministic(self):
        """Stable point ID is unaffected by model_version_id."""
        idx_a, s_a = _make_indexer(model_version_id="v1")
        idx_b, s_b = _make_indexer(model_version_id="v99")
        pa = _build_point(idx_a, s_a, video_id="v", segment_id=0, t_sec=1.0)
        pb = _build_point(idx_b, s_b, video_id="v", segment_id=0, t_sec=1.0)
        assert pa.id == pb.id

    def test_sup_style_version_string(self):
        """The 'sup_{job_id[:8]}' format used by the worker is stored verbatim."""
        version = "sup_1a2b3c4d"
        indexer, settings_mock = _make_indexer(model_version_id=version)
        point = _build_point(indexer, settings_mock)
        assert point.payload["model_version_id"] == version
