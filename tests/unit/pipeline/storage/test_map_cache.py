"""Unit tests for pipeline/map_cache.py."""

import json
import math
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from selfsuvis.pipeline.storage.map_cache import build_map_cache

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_point(
    clip: list[float],
    mission_id: str = "m1",
    t_sec: float = 1.0,
    gps: dict | None = None,
    enu: dict | None = None,
    robot_id: str | None = "robot_0",
    frame_path: str = "/data/frames/f.jpg",
) -> MagicMock:
    pt = MagicMock()
    pt.vector = {"clip": clip}
    pt.payload = {
        "type": "frame",
        "mission_id": mission_id,
        "t_sec": t_sec,
        "frame_path": frame_path,
        "robot_id": robot_id,
    }
    if gps is not None:
        pt.payload["gps"] = gps
    if enu is not None:
        pt.payload["enu"] = enu
    return pt


def _make_store(points: list[MagicMock]) -> MagicMock:
    store = MagicMock()
    store.collection_name = "test"
    # scroll returns (results, next_offset); single page with no next
    store.client.scroll.return_value = (points, None)
    return store


def _load_npz(raw: bytes) -> Any:
    return np.load(BytesIO(raw), allow_pickle=False)


# ── build_map_cache tests ─────────────────────────────────────────────────────


def test_empty_store_returns_valid_npz():
    store = _make_store([])
    raw = build_map_cache(store)
    cache = _load_npz(raw)
    assert "clip_vectors" in cache
    assert cache["clip_vectors"].shape == (0, 0)
    assert cache["gps"].shape == (0, 3)
    assert cache["enu"].shape == (0, 3)


def test_single_frame_vectors_correct():
    clip = [0.1, 0.2, 0.3]
    pt = _make_point(clip, gps={"lat": 48.0, "lon": 11.0, "alt": 100.0})
    store = _make_store([pt])
    cache = _load_npz(build_map_cache(store))

    assert cache["clip_vectors"].shape == (1, 3)
    np.testing.assert_allclose(cache["clip_vectors"][0], clip, atol=1e-5)


def test_gps_packed_correctly():
    pt = _make_point([0.0], gps={"lat": 48.123, "lon": 11.456, "alt": 200.0})
    cache = _load_npz(build_map_cache(_make_store([pt])))
    row = cache["gps"][0]
    assert row[0] == pytest.approx(48.123, abs=1e-4)
    assert row[1] == pytest.approx(11.456, abs=1e-4)
    assert row[2] == pytest.approx(200.0, abs=0.1)


def test_missing_gps_is_nan():
    pt = _make_point([0.0])  # no gps
    cache = _load_npz(build_map_cache(_make_store([pt])))
    assert math.isnan(cache["gps"][0, 0])
    assert math.isnan(cache["gps"][0, 1])


def test_enu_packed_correctly():
    pt = _make_point([0.0], enu={"tx": 10.0, "ty": 20.0, "tz": 5.0})
    cache = _load_npz(build_map_cache(_make_store([pt])))
    row = cache["enu"][0]
    assert row[0] == pytest.approx(10.0)
    assert row[1] == pytest.approx(20.0)
    assert row[2] == pytest.approx(5.0)


def test_missing_enu_is_nan():
    pt = _make_point([0.0])  # no enu
    cache = _load_npz(build_map_cache(_make_store([pt])))
    assert math.isnan(cache["enu"][0, 0])


def test_t_sec_packed():
    pt = _make_point([0.0], t_sec=42.5)
    cache = _load_npz(build_map_cache(_make_store([pt])))
    assert cache["t_sec"][0] == pytest.approx(42.5)


def test_meta_json_decodable():
    pt = _make_point(
        [0.0], mission_id="mission_abc", robot_id="robot_1", frame_path="/data/frames/x.jpg"
    )
    cache = _load_npz(build_map_cache(_make_store([pt])))
    meta = json.loads(bytes(cache["meta_json"]).decode())
    assert isinstance(meta, list)
    assert len(meta) == 1
    assert meta[0]["mission_id"] == "mission_abc"
    assert meta[0]["robot_id"] == "robot_1"
    assert meta[0]["frame_path"] == "/data/frames/x.jpg"


def test_multiple_frames_count():
    pts = [_make_point([float(i)] * 4) for i in range(5)]
    cache = _load_npz(build_map_cache(_make_store(pts)))
    assert cache["clip_vectors"].shape == (5, 4)
    assert cache["gps"].shape == (5, 3)
    assert cache["t_sec"].shape == (5,)


def test_mission_ids_filter_passed_to_scroll():
    store = _make_store([])
    build_map_cache(store, mission_ids=["m1", "m2"])
    call_kwargs = store.client.scroll.call_args[1]
    scroll_filter = call_kwargs["scroll_filter"]
    keys = [c.key for c in scroll_filter.must]
    assert "mission_id" in keys


def test_gps_bbox_filter_passed_to_scroll():
    store = _make_store([])
    build_map_cache(store, lat_min=47.0, lat_max=49.0, lon_min=10.0, lon_max=12.0)
    call_kwargs = store.client.scroll.call_args[1]
    keys = [c.key for c in call_kwargs["scroll_filter"].must]
    assert "gps.lat" in keys
    assert "gps.lon" in keys


def test_no_bbox_filter_skips_gps_conditions():
    store = _make_store([])
    build_map_cache(store)
    call_kwargs = store.client.scroll.call_args[1]
    keys = [c.key for c in call_kwargs["scroll_filter"].must]
    assert "gps.lat" not in keys


def test_pagination_exhausted():
    """scroll is called again when next_offset is returned."""
    store = MagicMock()
    store.collection_name = "test"
    pt = _make_point([1.0, 2.0])
    # First call returns a point + next_offset; second call returns empty
    store.client.scroll.side_effect = [
        ([pt], "some-offset"),
        ([], None),
    ]
    cache = _load_npz(build_map_cache(store))
    assert cache["clip_vectors"].shape[0] == 1
    assert store.client.scroll.call_count == 2


def test_frame_without_clip_skipped():
    """Points with no clip vector are silently skipped."""
    pt = MagicMock()
    pt.vector = {"dino": [0.1, 0.2]}  # no clip
    pt.payload = {"type": "frame", "t_sec": 1.0}
    store = _make_store([pt])
    cache = _load_npz(build_map_cache(store))
    assert cache["clip_vectors"].shape[0] == 0


def test_output_is_npz_bytes():
    raw = build_map_cache(_make_store([]))
    assert isinstance(raw, bytes)
    # NPZ files start with PK (zip magic)
    assert raw[:2] == b"PK"
