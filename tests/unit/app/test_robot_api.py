"""Unit tests for app.routers.robot — POST /query/pose."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Stub app.state before importing robot router to avoid live Qdrant connect ─
_mock_clip = MagicMock()
_mock_clip.embed_dim = 512
_mock_qdrant = MagicMock()
_mock_qdrant.collection_name = "test"

_state_stub = MagicMock()
_state_stub.clip_model = _mock_clip
_state_stub.qdrant_store = _mock_qdrant
sys.modules.setdefault("selfsuvis.app.state", _state_stub)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from selfsuvis.app.routers.robot import _enu_distance_m, _gps_distance_m  # noqa: E402
from selfsuvis.app.routers.robot import router as robot_router  # noqa: E402

_app = FastAPI()
_app.include_router(robot_router)
_client = TestClient(_app)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_qdrant_hit(lat, lon, mission_id="m1", t_sec=10.0, score=0.9):
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "gps": {"lat": lat, "lon": lon, "alt": 100.0},
        "mission_id": mission_id,
        "frame_id": f"f_{lat}_{lon}",
        "t_sec": t_sec,
        "frame_path": f"/data/frames/{mission_id}/frame.jpg",
        "global_pose_json": None,
    }
    return hit


def _query_response(hits: list) -> MagicMock:
    """Wrap a list of hits in a QueryResponse-like mock (has .points)."""
    resp = MagicMock()
    resp.points = hits
    return resp


def _mock_qdrant(hits: list):
    """Return a mock qdrant_store whose client.query_points yields the given hits."""
    mock_store = MagicMock()
    mock_store.collection_name = "test_collection"
    mock_store.client.query_points.return_value = _query_response(hits)
    return mock_store


def _mock_clip(dim=512):
    mock = MagicMock()
    mock.embed_dim = dim
    return mock


# ── _gps_distance_m tests ─────────────────────────────────────────────────────


def test_gps_distance_same_point():
    assert _gps_distance_m(48.0, 11.0, 48.0, 11.0) == pytest.approx(0.0)


def test_gps_distance_50m_north():
    dlat = 50.0 / 111_320.0
    d = _gps_distance_m(48.0, 11.0, 48.0 + dlat, 11.0)
    assert abs(d - 50.0) < 1.0  # within 1m


def test_gps_distance_symmetry():
    d1 = _gps_distance_m(48.0, 11.0, 48.001, 11.001)
    d2 = _gps_distance_m(48.001, 11.001, 48.0, 11.0)
    assert abs(d1 - d2) < 0.01


# ── POST /query/pose tests ────────────────────────────────────────────────────


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_returns_results(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    hits = [_make_qdrant_hit(48.0001, 11.0001), _make_qdrant_hit(48.0002, 11.0002)]
    mock_qdrant_store.client.query_points.return_value = _query_response(hits)

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0, "radius_m": 200.0, "top_k": 5},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_response_schema(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response(
        [_make_qdrant_hit(48.0001, 11.0001)]
    )

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0},
        headers={"X-API-Key": ""},
    )
    data = resp.json()
    assert data["query_lat"] == 48.0
    assert data["query_lon"] == 11.0
    assert "radius_m" in data
    assert "filter_strategy" in data


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_1d_filter_strategy(mock_clip, mock_qdrant_store, monkeypatch):
    from selfsuvis.pipeline.core import config

    monkeypatch.setattr(config.settings, "GPS_FILTER_2D", False)
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response([])

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0},
        headers={"X-API-Key": ""},
    )
    data = resp.json()
    assert data["filter_strategy"] == "1d+python"


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_2d_filter_strategy(mock_clip, mock_qdrant_store, monkeypatch):
    from selfsuvis.pipeline.core import config

    monkeypatch.setattr(config.settings, "GPS_FILTER_2D", True)
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response([])

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0},
        headers={"X-API-Key": ""},
    )
    data = resp.json()
    assert data["filter_strategy"] == "2d"


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_python_lon_postfilter(mock_clip, mock_qdrant_store, monkeypatch):
    """1D mode: hits outside lon bbox are post-filtered in Python."""
    from selfsuvis.pipeline.core import config

    monkeypatch.setattr(config.settings, "GPS_FILTER_2D", False)
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"

    # One hit inside bbox, one far outside (lon=99)
    inside = _make_qdrant_hit(48.0001, 11.0001)
    outside = _make_qdrant_hit(48.0001, 99.0)  # far outside lon bbox
    mock_qdrant_store.client.query_points.return_value = _query_response([inside, outside])

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0, "radius_m": 500.0, "top_k": 10},
        headers={"X-API-Key": ""},
    )
    data = resp.json()
    # Only the inside hit should survive post-filter
    assert len(data["results"]) == 1
    assert data["results"][0]["lon"] == pytest.approx(11.0001)


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_top_k_respected(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    hits = [_make_qdrant_hit(48.0 + i * 0.0001, 11.0) for i in range(10)]
    mock_qdrant_store.client.query_points.return_value = _query_response(hits)

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0, "top_k": 3},
        headers={"X-API-Key": ""},
    )
    data = resp.json()
    assert len(data["results"]) <= 3


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_empty_results(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response([])

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_qdrant_error_returns_503(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.side_effect = RuntimeError("connection refused")

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 503


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_distance_m_present(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response(
        [_make_qdrant_hit(48.0001, 11.0)]
    )

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0},
        headers={"X-API-Key": ""},
    )
    result = resp.json()["results"][0]
    assert result["distance_m"] is not None
    assert result["distance_m"] > 0


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_sorted_by_distance(mock_clip, mock_qdrant_store):
    """Results are sorted nearest-first."""
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    far = _make_qdrant_hit(48.01, 11.0, score=0.99)  # far but high score
    near = _make_qdrant_hit(48.0001, 11.0, score=0.5)  # near but low score
    mock_qdrant_store.client.query_points.return_value = _query_response([far, near])

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0, "top_k": 5},
        headers={"X-API-Key": ""},
    )
    dists = [r["distance_m"] for r in resp.json()["results"]]
    assert dists == sorted(dists)


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_pose_query_radius_validation(mock_clip, mock_qdrant_store):
    """radius_m < 1 should return 422."""
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response([])

    resp = _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0, "radius_m": 0.5},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 422


# ── _enu_distance_m tests ─────────────────────────────────────────────────────


def test_enu_distance_same_point():
    assert _enu_distance_m(10.0, 20.0, 5.0, 10.0, 20.0, 5.0) == pytest.approx(0.0)


def test_enu_distance_along_x():
    d = _enu_distance_m(0.0, 0.0, 0.0, 30.0, 0.0, 0.0)
    assert d == pytest.approx(30.0)


def test_enu_distance_3d():
    d = _enu_distance_m(0.0, 0.0, 0.0, 3.0, 4.0, 0.0)
    assert d == pytest.approx(5.0)


# ── ENU query path tests ──────────────────────────────────────────────────────


def _make_enu_hit(tx, ty, tz, mission_id="m1", t_sec=10.0, score=0.9):
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "enu": {"tx": tx, "ty": ty, "tz": tz},
        "mission_id": mission_id,
        "frame_id": f"f_{tx}_{ty}",
        "t_sec": t_sec,
        "frame_path": f"/data/frames/{mission_id}/frame.jpg",
        "global_pose_json": None,
    }
    return hit


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_enu_query_returns_results(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response(
        [_make_enu_hit(10.0, 20.0, 5.0)]
    )

    resp = _client.post(
        "/query/pose",
        json={"tx": 10.0, "ty": 20.0, "tz": 5.0, "radius_m": 50.0},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["filter_strategy"] == "enu+python"


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_enu_query_response_schema(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response([])

    resp = _client.post(
        "/query/pose",
        json={"tx": 0.0, "ty": 0.0, "tz": 0.0},
        headers={"X-API-Key": ""},
    )
    data = resp.json()
    assert data["query_tx"] == pytest.approx(0.0)
    assert data["query_ty"] == pytest.approx(0.0)
    assert data["query_tz"] == pytest.approx(0.0)
    assert data["query_lat"] is None
    assert data["query_lon"] is None


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_enu_query_3d_postfilter(mock_clip, mock_qdrant_store):
    """Hits outside 3D ENU sphere are excluded even if inside 2D bbox."""
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    # inside: 30m away in 3D
    inside = _make_enu_hit(30.0, 0.0, 0.0)
    # outside: 60m away — beyond radius_m=50
    outside = _make_enu_hit(60.0, 0.0, 0.0, mission_id="m2")
    mock_qdrant_store.client.query_points.return_value = _query_response([inside, outside])

    resp = _client.post(
        "/query/pose",
        json={"tx": 0.0, "ty": 0.0, "tz": 0.0, "radius_m": 50.0, "top_k": 10},
        headers={"X-API-Key": ""},
    )
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["distance_m"] == pytest.approx(30.0)


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_enu_query_distance_m_computed(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response(
        [_make_enu_hit(3.0, 4.0, 0.0)]
    )

    resp = _client.post(
        "/query/pose",
        json={"tx": 0.0, "ty": 0.0, "tz": 0.0, "radius_m": 50.0},
        headers={"X-API-Key": ""},
    )
    result = resp.json()["results"][0]
    assert result["distance_m"] == pytest.approx(5.0)


def test_pose_query_no_coords_returns_422():
    """Neither GPS nor ENU → 422."""
    resp = _client.post(
        "/query/pose",
        json={"radius_m": 50.0},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 422


def test_pose_query_partial_enu_returns_422():
    """tx+ty without tz is not enough for ENU path."""
    resp = _client.post(
        "/query/pose",
        json={"tx": 10.0, "ty": 20.0},
        headers={"X-API-Key": ""},
    )
    assert resp.status_code == 422


# ── robot_ids filter tests ────────────────────────────────────────────────────


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_robot_ids_filter_added_to_query(mock_clip, mock_qdrant_store):
    """robot_ids field causes a MatchAny condition on robot_id payload key."""
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response([])

    _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0, "robot_ids": ["robot_0", "robot_1"]},
        headers={"X-API-Key": ""},
    )
    call_kwargs = mock_qdrant_store.client.query_points.call_args[1]
    qf = call_kwargs["query_filter"]
    robot_keys = [c.key for c in qf.must if hasattr(c, "key")]
    assert "robot_id" in robot_keys


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_no_robot_ids_omits_robot_filter(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response([])

    _client.post(
        "/query/pose",
        json={"lat": 48.0, "lon": 11.0},
        headers={"X-API-Key": ""},
    )
    call_kwargs = mock_qdrant_store.client.query_points.call_args[1]
    qf = call_kwargs["query_filter"]
    robot_keys = [c.key for c in qf.must if hasattr(c, "key")]
    assert "robot_id" not in robot_keys


@patch("selfsuvis.app.routers.robot.qdrant_store")
@patch("selfsuvis.app.routers.robot.clip_model")
def test_robot_ids_filter_works_with_enu_path(mock_clip, mock_qdrant_store):
    mock_clip.embed_dim = 512
    mock_qdrant_store.collection_name = "test"
    mock_qdrant_store.client.query_points.return_value = _query_response([])

    _client.post(
        "/query/pose",
        json={"tx": 0.0, "ty": 0.0, "tz": 0.0, "robot_ids": ["drone_a"]},
        headers={"X-API-Key": ""},
    )
    call_kwargs = mock_qdrant_store.client.query_points.call_args[1]
    qf = call_kwargs["query_filter"]
    robot_keys = [c.key for c in qf.must if hasattr(c, "key")]
    assert "robot_id" in robot_keys
