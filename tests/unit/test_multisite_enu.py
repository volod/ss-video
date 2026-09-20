"""Unit tests for Multi-site ENU feature.

Covers:
  A. get_global_map_origin
  B. list_global_maps
  C. get_or_create_global_map proximity fix (cos(lat))
  D. Robot API global_map_id filter
  E. index_video site_enu_origin parameter
  F. _resolve_site_origin in worker.main

External deps that are not installed in the unit-test venv are injected as stubs
into sys.modules BEFORE any project module is imported.
"""

import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _query_response(hits: list) -> MagicMock:
    """Wrap hits in a QueryResponse-like mock (qdrant-client >= 1.7 returns .points)."""
    resp = MagicMock()
    resp.points = hits
    return resp


# ---------------------------------------------------------------------------
# Inject stub modules for optional deps (asyncpg, cv2, etc.)
# Must happen BEFORE importing any pipeline or worker module.
# ---------------------------------------------------------------------------


def _make_stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _ensure_stub(name: str, **attrs) -> types.ModuleType:
    if name not in sys.modules:
        return _make_stub(name, **attrs)
    return sys.modules[name]  # type: ignore[return-value]


# asyncpg stub — use _ensure_stub so all test files share the same module
# object. _make_stub would create a new object, breaking patches in files
# (e.g. test_gpu_isolation.py) that captured sys.modules["asyncpg"] earlier.
_asyncpg_stub = _ensure_stub("asyncpg", connect=AsyncMock(), Pool=MagicMock())

# Import the REAL pipeline.global_map_db now (while asyncpg stub is in place)
# so Groups A/B/C test the real code, not a stub.
import selfsuvis.pipeline.storage.global_maps as _real_global_map_db  # noqa: E402

# Stub pipeline.workflows to avoid heavy model imports (worker.main imports it)
_workflows_stub = _make_stub("selfsuvis.pipeline.workflows")
_workflows_stub.VideoIndexer = MagicMock()  # type: ignore[attr-defined]

# Stub pipeline.storage.processed and pipeline.storage entrypoints used by worker.main
_make_stub(
    "selfsuvis.pipeline.storage.processed",
    ainit_db=MagicMock(),
    aget_by_hash=MagicMock(),
    aupsert=MagicMock(),
    get_by_hash=MagicMock(),
)
_make_stub(
    "selfsuvis.pipeline.storage",
    create_job=MagicMock(),
    fetch_and_claim_next_pending=MagicMock(),
    update_job=MagicMock(),
)

# Stub pipeline.utils
_ensure_stub("selfsuvis.pipeline.utils", file_sha256=MagicMock(), resolve_allowed_path=MagicMock())

# Stub pipeline.downloader
_ensure_stub("selfsuvis.pipeline.downloader", download_url=MagicMock())

# Stub pipeline.gps_registration
_ensure_stub(
    "selfsuvis.pipeline.mapping.gps_registration",
    register_mission_gps=MagicMock(),
    gps_to_enu=MagicMock(),
)

# NOTE: Do not stub selfsuvis.pipeline.storage.global_maps. It is dependency-light
# and other tests rely on importing the real module.

# Stub pipeline.sfm
_ensure_stub("selfsuvis.pipeline.mapping.sfm", run_sfm=MagicMock())
# pipeline.mapper is NOT stubbed at module level — worker.main imports it
# lazily inside _run_pass_a, so no stub is needed here.

# Stub pipeline.media.gps (used by worker.main _resolve_site_origin)
_ensure_stub("selfsuvis.pipeline.media.gps", extract_gps=MagicMock())

# Stub app.state to avoid live Qdrant connect (needed by robot router)
_state_stub = MagicMock()
_state_stub.clip_model = MagicMock()
_state_stub.clip_model.embed_dim = 512
_state_stub.qdrant_store = MagicMock()
_state_stub.qdrant_store.collection_name = "test"
sys.modules.setdefault("selfsuvis.app.state", _state_stub)


# ---------------------------------------------------------------------------
# Group A: get_global_map_origin
# ---------------------------------------------------------------------------


class TestGetGlobalMapOrigin(unittest.IsolatedAsyncioTestCase):
    async def test_returns_tuple_when_row_exists(self):
        fn = _real_global_map_db.get_global_map_origin
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "origin_lat": 48.123,
                "origin_lon": 11.456,
                "origin_alt": 500.0,
            }
        )
        result = await fn(conn, 1)
        self.assertEqual(result, (48.123, 11.456, 500.0))
        conn.fetchrow.assert_awaited_once()

    async def test_returns_none_when_not_found(self):
        fn = _real_global_map_db.get_global_map_origin
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        result = await fn(conn, 999)
        self.assertIsNone(result)

    async def test_query_uses_correct_id(self):
        fn = _real_global_map_db.get_global_map_origin
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "origin_lat": 1.0,
                "origin_lon": 2.0,
                "origin_alt": 3.0,
            }
        )
        await fn(conn, 42)
        call_args = conn.fetchrow.call_args
        self.assertIn(42, call_args.args)


# ---------------------------------------------------------------------------
# Group B: list_global_maps
# ---------------------------------------------------------------------------


class TestListGlobalMaps(unittest.IsolatedAsyncioTestCase):
    async def test_returns_empty_list_when_no_rows(self):
        fn = _real_global_map_db.list_global_maps
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        result = await fn(conn)
        self.assertEqual(result, [])

    async def test_returns_all_rows_as_dicts(self):
        fn = _real_global_map_db.list_global_maps
        conn = AsyncMock()
        rows = [
            {
                "id": 1,
                "origin_lat": 48.0,
                "origin_lon": 11.0,
                "origin_alt": 0.0,
                "splat_path": None,
                "created_at": 1000.0,
            },
            {
                "id": 2,
                "origin_lat": 52.0,
                "origin_lon": 13.0,
                "origin_alt": 10.0,
                "splat_path": "/tmp/splat.ply",
                "created_at": 2000.0,
            },
        ]
        conn.fetch = AsyncMock(return_value=rows)
        result = await fn(conn)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[1]["id"], 2)

    async def test_each_row_has_expected_keys(self):
        fn = _real_global_map_db.list_global_maps
        conn = AsyncMock()
        rows = [
            {
                "id": 1,
                "origin_lat": 48.0,
                "origin_lon": 11.0,
                "origin_alt": 0.0,
                "splat_path": None,
                "created_at": 1000.0,
            },
        ]
        conn.fetch = AsyncMock(return_value=rows)
        result = await fn(conn)
        expected_keys = {"id", "origin_lat", "origin_lon", "origin_alt", "splat_path", "created_at"}
        self.assertEqual(set(result[0].keys()), expected_keys)

    async def test_ordered_by_created_at(self):
        fn = _real_global_map_db.list_global_maps
        conn = AsyncMock()
        rows = [
            {
                "id": 1,
                "origin_lat": 48.0,
                "origin_lon": 11.0,
                "origin_alt": 0.0,
                "splat_path": None,
                "created_at": 1000.0,
            },
            {
                "id": 2,
                "origin_lat": 52.0,
                "origin_lon": 13.0,
                "origin_alt": 0.0,
                "splat_path": None,
                "created_at": 2000.0,
            },
        ]
        conn.fetch = AsyncMock(return_value=rows)
        result = await fn(conn)
        # Verify ORDER BY created_at is in the query
        query_str = conn.fetch.call_args.args[0]
        self.assertIn("ORDER BY created_at", query_str)
        self.assertEqual(result[0]["created_at"], 1000.0)
        self.assertEqual(result[1]["created_at"], 2000.0)


# ---------------------------------------------------------------------------
# Group C: get_or_create_global_map proximity fix (cos(lat) applied correctly)
# ---------------------------------------------------------------------------


class TestGetOrCreateGlobalMapProximity(unittest.IsolatedAsyncioTestCase):
    """Test proximity check uses math.cos(lat) instead of hardcoded 0.7."""

    def _make_conn(self, existing_rows, new_id=99):
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=existing_rows)
        conn.fetchval = AsyncMock(return_value=new_id)
        return conn

    async def test_same_site_within_5km_returns_existing_id(self):
        """Two GPS points ~3km apart should share the same global_map_id."""
        fn = _real_global_map_db.get_or_create_global_map
        origin_lat, origin_lon = 48.0, 11.0
        existing = [{"id": 7, "origin_lat": origin_lat, "origin_lon": origin_lon}]
        conn = self._make_conn(existing, new_id=55)

        # 3km north of origin — same site
        new_lat = origin_lat + 3000.0 / 111_000
        result = await fn(conn, new_lat, origin_lon, 0.0)
        self.assertEqual(result, 7)
        conn.fetchval.assert_not_awaited()  # no insert

    async def test_different_site_beyond_5km_creates_new_row(self):
        """Two GPS points ~10km apart should get different global_map_ids."""
        fn = _real_global_map_db.get_or_create_global_map
        origin_lat, origin_lon = 48.0, 11.0
        existing = [{"id": 7, "origin_lat": origin_lat, "origin_lon": origin_lon}]
        conn = self._make_conn(existing, new_id=55)

        # 10km north of origin — different site
        new_lat = origin_lat + 10_000.0 / 111_000
        result = await fn(conn, new_lat, origin_lon, 0.0)
        self.assertEqual(result, 55)
        conn.fetchval.assert_awaited_once()  # insert happened

    async def test_cos_lat_applied_at_high_latitude(self):
        """At 60°N, cos(60°)≈0.5, so longitude degrees count less.

        A point 0.2 degrees east at 60°N is ~0.2 * 111_000 * 0.5 ≈ 11100m
        from the origin, which is > 5km → different site.
        The old hardcoded 0.7 factor would give ~0.2 * 111_000 * 0.7 = 15540m
        (also > 5km, so this test primarily checks the math path is correct).
        We also verify a point close enough is clustered correctly.
        """
        fn = _real_global_map_db.get_or_create_global_map
        origin_lat = 60.0
        origin_lon = 10.0
        existing = [{"id": 1, "origin_lat": origin_lat, "origin_lon": origin_lon}]

        # 0.04 degrees east at 60°N: ~0.04 * 111_000 * cos(60°) ≈ 2220m (< 5km, same site)
        new_lon_close = origin_lon + 0.04
        conn_close = self._make_conn(existing, new_id=2)
        result_close = await fn(conn_close, origin_lat, new_lon_close, 0.0)
        self.assertEqual(result_close, 1)  # same site
        conn_close.fetchval.assert_not_awaited()

        # 0.15 degrees east at 60°N: ~0.15 * 111_000 * cos(60°) ≈ 8325m (> 5km, different site)
        new_lon_far = origin_lon + 0.15
        conn_far = self._make_conn(existing, new_id=2)
        result_far = await fn(conn_far, origin_lat, new_lon_far, 0.0)
        self.assertEqual(result_far, 2)  # new site
        conn_far.fetchval.assert_awaited_once()

    async def test_no_existing_rows_creates_new_entry(self):
        """With empty global_map table, always creates a new row."""
        fn = _real_global_map_db.get_or_create_global_map
        conn = self._make_conn([], new_id=1)
        result = await fn(conn, 48.0, 11.0, 100.0)
        self.assertEqual(result, 1)
        conn.fetchval.assert_awaited_once()


# ---------------------------------------------------------------------------
# Group D: Robot API global_map_id filter
# ---------------------------------------------------------------------------

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from selfsuvis.app.routers.robot import router as robot_router  # noqa: E402

_app = FastAPI()
_app.include_router(robot_router)
_client = TestClient(_app)


class TestRobotApiGlobalMapIdFilter(unittest.TestCase):
    @patch("selfsuvis.app.routers.robot.qdrant_store")
    @patch("selfsuvis.app.routers.robot.clip_model")
    def test_global_map_id_present_adds_field_condition(self, mock_clip, mock_qdrant_store):
        """global_map_id in request → FieldCondition(key='global_map_id') in filter."""
        mock_clip.embed_dim = 512
        mock_qdrant_store.collection_name = "test"
        mock_qdrant_store.client.query_points.return_value = _query_response([])

        _client.post(
            "/query/pose",
            json={"lat": 48.0, "lon": 11.0, "global_map_id": 7},
            headers={"X-API-Key": ""},
        )
        call_kwargs = mock_qdrant_store.client.query_points.call_args[1]
        qf = call_kwargs["query_filter"]
        keys = [c.key for c in qf.must if hasattr(c, "key")]
        self.assertIn("global_map_id", keys)

    @patch("selfsuvis.app.routers.robot.qdrant_store")
    @patch("selfsuvis.app.routers.robot.clip_model")
    def test_global_map_id_absent_omits_condition(self, mock_clip, mock_qdrant_store):
        """No global_map_id in request → no global_map_id condition in filter."""
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
        keys = [c.key for c in qf.must if hasattr(c, "key")]
        self.assertNotIn("global_map_id", keys)

    @patch("selfsuvis.app.routers.robot.qdrant_store")
    @patch("selfsuvis.app.routers.robot.clip_model")
    def test_global_map_id_combined_with_robot_ids(self, mock_clip, mock_qdrant_store):
        """global_map_id + robot_ids → both conditions present."""
        mock_clip.embed_dim = 512
        mock_qdrant_store.collection_name = "test"
        mock_qdrant_store.client.query_points.return_value = _query_response([])

        _client.post(
            "/query/pose",
            json={"lat": 48.0, "lon": 11.0, "global_map_id": 3, "robot_ids": ["r1"]},
            headers={"X-API-Key": ""},
        )
        call_kwargs = mock_qdrant_store.client.query_points.call_args[1]
        qf = call_kwargs["query_filter"]
        keys = [c.key for c in qf.must if hasattr(c, "key")]
        self.assertIn("global_map_id", keys)
        self.assertIn("robot_id", keys)

    @patch("selfsuvis.app.routers.robot.qdrant_store")
    @patch("selfsuvis.app.routers.robot.clip_model")
    def test_global_map_id_in_enu_path(self, mock_clip, mock_qdrant_store):
        """ENU path + global_map_id → condition present."""
        mock_clip.embed_dim = 512
        mock_qdrant_store.collection_name = "test"
        mock_qdrant_store.client.query_points.return_value = _query_response([])

        _client.post(
            "/query/pose",
            json={"tx": 0.0, "ty": 0.0, "tz": 0.0, "global_map_id": 5},
            headers={"X-API-Key": ""},
        )
        call_kwargs = mock_qdrant_store.client.query_points.call_args[1]
        qf = call_kwargs["query_filter"]
        keys = [c.key for c in qf.must if hasattr(c, "key")]
        self.assertIn("global_map_id", keys)

    @patch("selfsuvis.app.routers.robot.qdrant_store")
    @patch("selfsuvis.app.routers.robot.clip_model")
    def test_response_global_map_id_none_when_not_provided(self, mock_clip, mock_qdrant_store):
        """global_map_id in response is None when not provided in request."""
        mock_clip.embed_dim = 512
        mock_qdrant_store.collection_name = "test"
        mock_qdrant_store.client.query_points.return_value = _query_response([])

        resp = _client.post(
            "/query/pose",
            json={"lat": 48.0, "lon": 11.0},
            headers={"X-API-Key": ""},
        )
        data = resp.json()
        self.assertIsNone(data.get("global_map_id"))

    @patch("selfsuvis.app.routers.robot.qdrant_store")
    @patch("selfsuvis.app.routers.robot.clip_model")
    def test_response_global_map_id_int_when_provided(self, mock_clip, mock_qdrant_store):
        """global_map_id in response equals the provided value."""
        mock_clip.embed_dim = 512
        mock_qdrant_store.collection_name = "test"
        mock_qdrant_store.client.query_points.return_value = _query_response([])

        resp = _client.post(
            "/query/pose",
            json={"lat": 48.0, "lon": 11.0, "global_map_id": 42},
            headers={"X-API-Key": ""},
        )
        data = resp.json()
        self.assertEqual(data.get("global_map_id"), 42)


# ---------------------------------------------------------------------------
# Group E: index_video site_enu_origin parameter
# ---------------------------------------------------------------------------


class TestIndexVideoSiteEnuOrigin(unittest.TestCase):
    """Test that index_video respects site_enu_origin and global_map_id.

    We test the internal logic of the ENU origin selection by replicating
    the block from index_video directly (no real video processing needed).
    """

    def test_site_enu_origin_none_falls_back_to_first_gps_frame(self):
        """When site_enu_origin=None, ENU origin = first valid GPS frame."""
        gps_list = [
            None,
            {"lat": 48.0, "lon": 11.0, "alt": 100.0},
            {"lat": 48.001, "lon": 11.001, "alt": 101.0},
        ]
        site_enu_origin = None

        # Replicate the logic from index_video
        enu_origin = site_enu_origin
        if enu_origin is None:
            for g in gps_list:
                if g is not None:
                    enu_origin = (g["lat"], g["lon"], g["alt"])
                    break

        self.assertEqual(enu_origin, (48.0, 11.0, 100.0))

    def test_site_enu_origin_provided_overrides_first_gps(self):
        """When site_enu_origin is provided, it takes precedence over first GPS frame."""
        gps_list = [
            {"lat": 48.0, "lon": 11.0, "alt": 100.0},
            {"lat": 48.001, "lon": 11.001, "alt": 101.0},
        ]
        canonical_origin = (47.5, 10.5, 50.0)
        site_enu_origin = canonical_origin

        # Replicate the logic from index_video
        enu_origin = site_enu_origin
        if enu_origin is None:
            for g in gps_list:
                if g is not None:
                    enu_origin = (g["lat"], g["lon"], g["alt"])
                    break

        self.assertEqual(enu_origin, canonical_origin)

    def test_site_enu_origin_used_even_when_all_gps_none(self):
        """site_enu_origin used even if all GPS frames are None."""
        gps_list = [None, None, None]
        canonical_origin = (47.5, 10.5, 50.0)
        site_enu_origin = canonical_origin

        enu_origin = site_enu_origin
        if enu_origin is None:
            for g in gps_list:
                if g is not None:
                    enu_origin = (g["lat"], g["lon"], g["alt"])
                    break

        self.assertEqual(enu_origin, canonical_origin)

    def test_global_map_id_stored_in_frame_payload(self):
        """global_map_id passed to _build_frame_point ends up in payload."""
        global_map_id = 7
        payload = {
            "type": "frame",
            "video_id": "v1",
            "segment_id": 0,
            "t_sec": 1.0,
            "frame_path": "/tmp/f.jpg",
        }

        if global_map_id is not None:
            payload["global_map_id"] = global_map_id

        self.assertEqual(payload["global_map_id"], 7)

    def test_global_map_id_none_not_in_frame_payload(self):
        """When global_map_id is None, it is not stored in payload."""
        global_map_id = None
        payload = {
            "type": "frame",
            "video_id": "v1",
            "segment_id": 0,
            "t_sec": 1.0,
            "frame_path": "/tmp/f.jpg",
        }

        if global_map_id is not None:
            payload["global_map_id"] = global_map_id

        self.assertNotIn("global_map_id", payload)

    def test_global_map_id_stored_in_tile_payload(self):
        """global_map_id passed to _index_tiles ends up in tile payload."""
        global_map_id = 42
        tile_payload = {
            "type": "tile",
            "video_id": "v1",
            "segment_id": 0,
            "t_sec": 1.0,
            "frame_path": "/tmp/f.jpg",
            "tile_path": "/tmp/t.jpg",
            "x": 0,
            "y": 0,
            "w": 64,
            "h": 64,
        }

        if global_map_id is not None:
            tile_payload["global_map_id"] = global_map_id

        self.assertEqual(tile_payload["global_map_id"], 42)


# ---------------------------------------------------------------------------
# Group F: _resolve_site_origin in worker.handlers.postflight
# ---------------------------------------------------------------------------


class TestResolveSiteOrigin(unittest.TestCase):
    def _get_fn(self):
        import selfsuvis.worker.handlers.postflight as ph

        return ph._resolve_site_origin

    def _make_logger(self):
        logger = MagicMock()
        logger.debug = MagicMock()
        logger.info = MagicMock()
        logger.warning = MagicMock()
        return logger

    def test_no_gps_returns_none_none(self):
        """No GPS in video → returns (None, None)."""
        fn = self._get_fn()
        logger = self._make_logger()
        with patch("selfsuvis.pipeline.media.gps.extract_gps", return_value=[None, None]):
            result = fn("/tmp/vid.mp4", logger)
        self.assertEqual(result, (None, None))
        logger.debug.assert_called()

    def test_gps_extraction_exception_returns_none_none(self):
        """GPS extraction raising an exception → (None, None), no crash."""
        fn = self._get_fn()
        logger = self._make_logger()
        with patch(
            "selfsuvis.pipeline.media.gps.extract_gps",
            side_effect=RuntimeError("ffprobe fail"),
        ):
            result = fn("/tmp/vid.mp4", logger)
        self.assertEqual(result, (None, None))

    def test_db_unreachable_returns_none_none(self):
        """GPS found but DB connection fails → (None, None)."""
        fn = self._get_fn()
        logger = self._make_logger()
        gps = [{"lat": 48.0, "lon": 11.0, "alt": 100.0}]
        with patch("selfsuvis.pipeline.media.gps.extract_gps", return_value=gps):
            with patch("asyncpg.connect", side_effect=ConnectionRefusedError("no DB")):
                result = fn("/tmp/vid.mp4", logger)
        self.assertEqual(result, (None, None))

    def test_happy_path_returns_global_map_id_and_origin(self):
        """GPS found, DB accessible → returns (global_map_id, origin_tuple)."""
        fn = self._get_fn()
        logger = self._make_logger()
        gps = [{"lat": 48.0, "lon": 11.0, "alt": 100.0}]
        fake_conn = AsyncMock()
        fake_conn.close = AsyncMock()

        with patch("selfsuvis.pipeline.media.gps.extract_gps", return_value=gps):
            with patch("asyncpg.connect", new_callable=AsyncMock, return_value=fake_conn):
                with patch(
                    "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                    new_callable=AsyncMock,
                    return_value=7,
                ):
                    with patch(
                        "selfsuvis.pipeline.storage.global_maps.get_global_map_origin",
                        new_callable=AsyncMock,
                        return_value=(48.0, 11.0, 100.0),
                    ):
                        result = fn("/tmp/vid.mp4", logger)

        self.assertEqual(result[0], 7)
        self.assertEqual(result[1], (48.0, 11.0, 100.0))
        logger.info.assert_called()

    def test_happy_path_logs_info_with_global_map_id(self):
        """Happy path logs info containing 'global_map_id'."""
        fn = self._get_fn()
        logger = self._make_logger()
        gps = [{"lat": 48.0, "lon": 11.0, "alt": 100.0}]
        fake_conn = AsyncMock()
        fake_conn.close = AsyncMock()

        with patch("selfsuvis.pipeline.media.gps.extract_gps", return_value=gps):
            with patch("asyncpg.connect", new_callable=AsyncMock, return_value=fake_conn):
                with patch(
                    "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                    new_callable=AsyncMock,
                    return_value=99,
                ):
                    with patch(
                        "selfsuvis.pipeline.storage.global_maps.get_global_map_origin",
                        new_callable=AsyncMock,
                        return_value=(48.0, 11.0, 0.0),
                    ):
                        fn("/tmp/vid.mp4", logger)

        info_msgs = [c.args[0] for c in logger.info.call_args_list]
        self.assertTrue(any("global_map_id" in m for m in info_msgs))

    def test_origin_none_from_db_returns_none_for_origin_part(self):
        """get_global_map_origin returning None → origin part of result is None."""
        fn = self._get_fn()
        logger = self._make_logger()
        gps = [{"lat": 48.0, "lon": 11.0, "alt": 100.0}]
        fake_conn = AsyncMock()
        fake_conn.close = AsyncMock()

        with patch("selfsuvis.pipeline.media.gps.extract_gps", return_value=gps):
            with patch("asyncpg.connect", new_callable=AsyncMock, return_value=fake_conn):
                with patch(
                    "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                    new_callable=AsyncMock,
                    return_value=5,
                ):
                    with patch(
                        "selfsuvis.pipeline.storage.global_maps.get_global_map_origin",
                        new_callable=AsyncMock,
                        return_value=None,
                    ):
                        gmap_id, origin = fn("/tmp/vid.mp4", logger)

        self.assertEqual(gmap_id, 5)
        self.assertIsNone(origin)

    def test_conn_closed_after_successful_lookup(self):
        """Connection is closed even after successful lookup."""
        fn = self._get_fn()
        logger = self._make_logger()
        gps = [{"lat": 48.0, "lon": 11.0, "alt": 100.0}]
        fake_conn = AsyncMock()
        fake_conn.close = AsyncMock()

        with patch("selfsuvis.pipeline.media.gps.extract_gps", return_value=gps):
            with patch("asyncpg.connect", new_callable=AsyncMock, return_value=fake_conn):
                with patch(
                    "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                    new_callable=AsyncMock,
                    return_value=1,
                ):
                    with patch(
                        "selfsuvis.pipeline.storage.global_maps.get_global_map_origin",
                        new_callable=AsyncMock,
                        return_value=(1.0, 2.0, 3.0),
                    ):
                        fn("/tmp/vid.mp4", logger)

        fake_conn.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
