"""Unit tests for Phase 5 last_visits extension in POST /query/pose.

Tests _get_last_visits() and verifies it appears in PoseQueryResponse.
No live Qdrant or PostgreSQL required.
"""

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub out app.state before any router import to avoid torchvision circular import
_state_stub = MagicMock()
_state_stub.clip_model = MagicMock()
_state_stub.qdrant_store = MagicMock()
_state_stub.dino_model = None
sys.modules.setdefault("selfsuvis.app.state", _state_stub)


# ── _get_last_visits ──────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_get_last_visits_returns_matching_rows():
    from selfsuvis.app.routers.robot import _get_last_visits

    now = datetime.now(timezone.utc)
    fake_rows = [
        {
            "mission_id": "m1",
            "frame_id": "f1",
            "t_sec": 10.0,
            "gps_lat": 47.0,
            "gps_lon": 8.0,
            "caption": "Two trucks.",
            "facts_json": {
                "vehicle_groups": [{"count": 2}],
                "road_condition": "clear",
            },
            "created_at": now,
        },
    ]

    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=fake_rows)

    visits = await _get_last_visits(pool, lat=47.0, lon=8.0, radius_m=50.0, limit=3)

    assert len(visits) == 1
    v = visits[0]
    assert v.mission_id == "m1"
    assert v.frame_id == "f1"
    assert v.caption == "Two trucks."
    assert v.vehicle_count == 2
    assert v.road_condition == "clear"
    assert v.lat == pytest.approx(47.0)
    assert v.lon == pytest.approx(8.0)


@pytest.mark.anyio
async def test_get_last_visits_counts_vehicles_across_groups():
    from selfsuvis.app.routers.robot import _get_last_visits

    now = datetime.now(timezone.utc)
    fake_rows = [
        {
            "mission_id": "m1",
            "frame_id": "f1",
            "t_sec": 5.0,
            "gps_lat": 47.0,
            "gps_lon": 8.0,
            "caption": None,
            "facts_json": {
                "vehicle_groups": [{"count": 3}, {"count": 2}],
            },
            "created_at": now,
        }
    ]
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=fake_rows)

    visits = await _get_last_visits(pool, lat=47.0, lon=8.0, radius_m=50.0)
    assert visits[0].vehicle_count == 5


@pytest.mark.anyio
async def test_get_last_visits_returns_empty_on_db_error():
    from selfsuvis.app.routers.robot import _get_last_visits

    pool = MagicMock()
    pool.fetch = AsyncMock(side_effect=Exception("relation scene_timeline does not exist"))

    visits = await _get_last_visits(pool, lat=47.0, lon=8.0, radius_m=50.0)
    assert visits == []


@pytest.mark.anyio
async def test_get_last_visits_returns_empty_when_no_rows():
    from selfsuvis.app.routers.robot import _get_last_visits

    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])

    visits = await _get_last_visits(pool, lat=47.0, lon=8.0, radius_m=50.0)
    assert visits == []


@pytest.mark.anyio
async def test_get_last_visits_null_facts_json():
    from selfsuvis.app.routers.robot import _get_last_visits

    now = datetime.now(timezone.utc)
    fake_rows = [
        {
            "mission_id": "m2",
            "frame_id": "f2",
            "t_sec": 20.0,
            "gps_lat": 47.1,
            "gps_lon": 8.1,
            "caption": "Empty road.",
            "facts_json": None,
            "created_at": now,
        }
    ]
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=fake_rows)

    visits = await _get_last_visits(pool, lat=47.1, lon=8.1, radius_m=50.0)
    assert visits[0].vehicle_count is None
    assert visits[0].road_condition is None


# ── PoseQueryResponse includes last_visits ────────────────────────────────────


def test_pose_query_response_has_last_visits_field():
    """PoseQueryResponse accepts last_visits=None (default) and a list."""
    from selfsuvis.app.routers.robot import PoseQueryResponse, TimelineVisit

    resp_none = PoseQueryResponse(
        results=[],
        query_lat=47.0,
        query_lon=8.0,
        query_tx=None,
        query_ty=None,
        query_tz=None,
        radius_m=50.0,
        filter_strategy="1d+python",
        global_map_id=None,
        last_visits=None,
    )
    assert resp_none.last_visits is None

    visit = TimelineVisit(
        mission_id="m1",
        frame_id="f1",
        t_sec=5.0,
        lat=47.0,
        lon=8.0,
        caption="Test.",
        road_condition="clear",
        vehicle_count=3,
        created_at="2026-04-06T12:00:00+00:00",
    )
    resp_with = PoseQueryResponse(
        results=[],
        query_lat=47.0,
        query_lon=8.0,
        query_tx=None,
        query_ty=None,
        query_tz=None,
        radius_m=50.0,
        filter_strategy="1d+python",
        global_map_id=None,
        last_visits=[visit],
    )
    assert resp_with.last_visits is not None
    assert len(resp_with.last_visits) == 1
    assert resp_with.last_visits[0].vehicle_count == 3


def test_timeline_visit_model_validation():
    """TimelineVisit allows all-None optional fields."""
    from selfsuvis.app.routers.robot import TimelineVisit

    v = TimelineVisit(
        mission_id="m1",
        frame_id="f1",
        t_sec=None,
        lat=None,
        lon=None,
        caption=None,
        road_condition=None,
        vehicle_count=None,
        created_at=None,
    )
    assert v.mission_id == "m1"
    assert v.vehicle_count is None
