"""Unit tests for ICP fusion integration in pipeline.mapper and pipeline.global_map_db.

Uses unittest.mock to stub HTTP calls and asyncpg connections; no live services
or open3d required.
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

# ── MockConn for asyncpg tests ────────────────────────────────────────────────


class _Row(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class MockConn:
    """Minimal asyncpg Connection mock backed by in-memory dicts."""

    def __init__(self):
        self._global_maps: list = []  # rows in global_map
        self._gmm: list = []  # rows in global_map_missions
        self._missions: list = []  # rows in missions (for join)
        self._next_id = 1

    def _new_id(self):
        val = self._next_id
        self._next_id += 1
        return val

    async def fetchval(self, query: str, *args) -> Any:
        q = query.strip().upper()
        if "INSERT INTO GLOBAL_MAP" in q:
            origin_lat, origin_lon, origin_alt, created_at, updated_at = args
            row = _Row(
                id=self._new_id(),
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                origin_alt=origin_alt,
                splat_path=None,
                created_at=created_at,
                updated_at=updated_at,
            )
            self._global_maps.append(row)
            return row["id"]
        raise NotImplementedError(f"fetchval not handled: {query[:60]}")

    async def fetch(self, query: str, *args) -> list:
        q = query.strip().upper()
        if "FROM GLOBAL_MAP ORDER" in q:
            return list(self._global_maps)
        if "FROM GLOBAL_MAP_MISSIONS" in q and "JOIN MISSIONS" in q:
            # Return splat_path of registered missions with splat_path set
            result = []
            gm_id = args[0]
            for gmm in self._gmm:
                if gmm["global_map_id"] == gm_id:
                    for m in self._missions:
                        if m["id"] == gmm["mission_id"] and m.get("splat_path"):
                            result.append(_Row(splat_path=m["splat_path"]))
            return result
        if "FROM GLOBAL_MAP_MISSIONS" in q:
            gm_id = args[0]
            return [_Row(**r) for r in self._gmm if r["global_map_id"] == gm_id]
        raise NotImplementedError(f"fetch not handled: {query[:60]}")

    async def fetchrow(self, query: str, *args) -> _Row | None:
        q = query.strip().upper()
        if "FROM GLOBAL_MAP WHERE ID" in q:
            gm_id = args[0]
            for row in self._global_maps:
                if row["id"] == gm_id:
                    return row
            return None
        raise NotImplementedError(f"fetchrow not handled: {query[:60]}")

    async def execute(self, query: str, *args) -> None:
        q = query.strip().upper()
        if "INSERT INTO GLOBAL_MAP_MISSIONS" in q:
            gm_id, mission_id, transform_json, reg_error, registered_at = args
            # Upsert: remove existing if conflict
            self._gmm = [
                r
                for r in self._gmm
                if not (r["global_map_id"] == gm_id and r["mission_id"] == mission_id)
            ]
            self._gmm.append(
                {
                    "global_map_id": gm_id,
                    "mission_id": mission_id,
                    "registration_transform_json": transform_json,
                    "registration_error": reg_error,
                    "registered_at": registered_at,
                }
            )
        elif "UPDATE GLOBAL_MAP SET SPLAT_PATH" in q:
            splat_path, updated_at, gm_id = args
            for row in self._global_maps:
                if row["id"] == gm_id:
                    row["splat_path"] = splat_path
                    row["updated_at"] = updated_at
        else:
            raise NotImplementedError(f"execute not handled: {query[:60]}")


# ── global_map_db tests ───────────────────────────────────────────────────────


class TestGetOrCreateGlobalMap:
    def test_creates_new_when_empty(self):
        from selfsuvis.pipeline.storage.global_maps import get_or_create_global_map

        conn = MockConn()
        gm_id = asyncio.run(get_or_create_global_map(conn, 48.0, 11.0, 100.0))
        assert gm_id == 1
        assert len(conn._global_maps) == 1

    def test_returns_existing_for_nearby_origin(self):
        from selfsuvis.pipeline.storage.global_maps import get_or_create_global_map

        conn = MockConn()
        id1 = asyncio.run(get_or_create_global_map(conn, 48.0, 11.0, 100.0))
        # 0.001 deg ≈ 111m — within 5km proximity threshold
        id2 = asyncio.run(get_or_create_global_map(conn, 48.001, 11.001, 100.0))
        assert id1 == id2
        assert len(conn._global_maps) == 1

    def test_creates_new_for_distant_origin(self):
        from selfsuvis.pipeline.storage.global_maps import get_or_create_global_map

        conn = MockConn()
        id1 = asyncio.run(get_or_create_global_map(conn, 48.0, 11.0, 100.0))
        # 0.1 deg ≈ 11km — beyond 5km threshold
        id2 = asyncio.run(get_or_create_global_map(conn, 48.1, 11.1, 100.0))
        assert id1 != id2
        assert len(conn._global_maps) == 2


class TestGetGlobalMapSplats:
    def test_empty_when_no_missions_registered(self):
        from selfsuvis.pipeline.storage.global_maps import get_global_map_splats

        conn = MockConn()
        conn._global_maps.append(_Row(id=1, origin_lat=48.0, origin_lon=11.0))
        paths = asyncio.run(get_global_map_splats(conn, 1))
        assert paths == []

    def test_returns_splat_paths_for_registered_missions(self):
        from selfsuvis.pipeline.storage.global_maps import get_global_map_splats

        conn = MockConn()
        conn._missions = [
            {"id": "m1", "splat_path": "/data/maps/m1/splat.ply"},
            {"id": "m2", "splat_path": None},
        ]
        conn._gmm = [
            {
                "global_map_id": 1,
                "mission_id": "m1",
                "registration_transform_json": "[]",
                "registration_error": None,
                "registered_at": time.time(),
            },
            {
                "global_map_id": 1,
                "mission_id": "m2",
                "registration_transform_json": "[]",
                "registration_error": None,
                "registered_at": time.time(),
            },
        ]
        paths = asyncio.run(get_global_map_splats(conn, 1))
        assert paths == ["/data/maps/m1/splat.ply"]


class TestRegisterMission:
    def test_inserts_new_registration(self):
        from selfsuvis.pipeline.storage.global_maps import register_mission

        conn = MockConn()
        transform = [[1, 0, 0, 7], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        asyncio.run(register_mission(conn, 1, "mission-abc", transform, 0.042))
        assert len(conn._gmm) == 1
        row = conn._gmm[0]
        assert row["mission_id"] == "mission-abc"
        assert row["global_map_id"] == 1
        assert json.loads(row["registration_transform_json"]) == transform
        assert abs(row["registration_error"] - 0.042) < 1e-6

    def test_upserts_on_conflict(self):
        from selfsuvis.pipeline.storage.global_maps import register_mission

        conn = MockConn()
        t1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        t2 = [[1, 0, 0, 7], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        asyncio.run(register_mission(conn, 1, "mission-abc", t1, None))
        asyncio.run(register_mission(conn, 1, "mission-abc", t2, 0.05))
        assert len(conn._gmm) == 1
        assert json.loads(conn._gmm[0]["registration_transform_json"]) == t2
        assert abs(conn._gmm[0]["registration_error"] - 0.05) < 1e-6

    def test_phase1_allows_none_error(self):
        from selfsuvis.pipeline.storage.global_maps import register_mission

        conn = MockConn()
        asyncio.run(register_mission(conn, 1, "mission-abc", [[1, 0, 0, 0]] * 4, None))
        assert conn._gmm[0]["registration_error"] is None


class TestUpdateGlobalMapSplat:
    def test_sets_splat_path(self):
        from selfsuvis.pipeline.storage.global_maps import update_global_map_splat

        conn = MockConn()
        conn._global_maps.append(_Row(id=1, splat_path=None, updated_at=0.0))
        asyncio.run(update_global_map_splat(conn, 1, "/data/maps/fused.ply"))
        assert conn._global_maps[0]["splat_path"] == "/data/maps/fused.ply"


class TestGetGlobalMapById:
    def test_returns_row(self):
        from selfsuvis.pipeline.storage.global_maps import get_global_map_by_id

        conn = MockConn()
        conn._global_maps.append(_Row(id=1, origin_lat=48.0, origin_lon=11.0))
        row = asyncio.run(get_global_map_by_id(conn, 1))
        assert row is not None
        assert row["origin_lat"] == 48.0

    def test_returns_none_when_missing(self):
        from selfsuvis.pipeline.storage.global_maps import get_global_map_by_id

        conn = MockConn()
        row = asyncio.run(get_global_map_by_id(conn, 999))
        assert row is None


# ── _call_icp_fuse tests (pipeline/mapper.py) ─────────────────────────────────


class TestCallIcpFuse:
    def test_returns_dict_on_success(self):
        from selfsuvis.pipeline.mapping.mapper import _call_icp_fuse

        mock_resp = {
            "status": "ok",
            "transform_4x4": [[1, 0, 0, 7], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "rmse": 0.05,
            "fitness": 0.85,
            "converged": True,
            "message": "",
        }
        with patch("selfsuvis.pipeline.mapping.mapper.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_resp
            mock_post.return_value.raise_for_status = MagicMock()
            result = _call_icp_fuse("/src.ply", "/tgt.ply")
        assert result is not None
        assert result["status"] == "ok"
        assert result["converged"] is True

    def test_returns_none_on_connection_error(self):
        import requests as req_lib

        from selfsuvis.pipeline.mapping.mapper import _call_icp_fuse

        with patch(
            "selfsuvis.pipeline.mapping.mapper.requests.post",
            side_effect=req_lib.exceptions.ConnectionError("refused"),
        ):
            result = _call_icp_fuse("/src.ply", "/tgt.ply")
        assert result is None

    def test_passes_source_and_target_path(self):
        from selfsuvis.pipeline.mapping.mapper import _call_icp_fuse

        with patch("selfsuvis.pipeline.mapping.mapper.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "status": "no_overlap",
                "converged": False,
                "message": "",
            }
            mock_post.return_value.raise_for_status = MagicMock()
            _call_icp_fuse("/src.ply", "/tgt.ply")
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["source_path"] == "/src.ply"
        assert payload["target_path"] == "/tgt.ply"

    def test_passes_optional_meta(self):
        from selfsuvis.pipeline.mapping.mapper import _call_icp_fuse

        src_meta = {"origin_lat": 48.0, "origin_lon": 11.0, "origin_alt": 100.0}
        with patch("selfsuvis.pipeline.mapping.mapper.requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "status": "ok",
                "converged": True,
                "message": "",
            }
            mock_post.return_value.raise_for_status = MagicMock()
            _call_icp_fuse("/src.ply", "/tgt.ply", source_meta=src_meta)
        payload = mock_post.call_args[1]["json"]
        assert payload.get("source_meta") == src_meta


# ── run_mapper ICP integration tests ──────────────────────────────────────────


def _make_sfm_results(n: int = 35) -> list[dict[str, Any]]:
    """Return a minimal list of SfM results with pose_status=success."""
    return [
        {
            "frame_path": f"/frames/f{i}.jpg",
            "pose_json": "{}",
            "pose_status": "success",
            "scene_index": 0,
        }
        for i in range(n)
    ]


class TestRunMapperIcpIntegration:
    def test_icp_results_empty_when_no_targets(self):
        from selfsuvis.pipeline.mapping.mapper import run_mapper

        with patch("selfsuvis.pipeline.mapping.mapper._train_scene") as mock_train:
            mock_train.return_value = {
                "map_status": "success",
                "splat_path": "/maps/m1/splat.ply",
                "message": "done",
            }
            result = run_mapper("m1", _make_sfm_results(), target_splat_paths=None)
        assert result["icp_results"] == []

    def test_icp_results_populated_when_target_provided(self):
        from selfsuvis.pipeline.mapping.mapper import run_mapper

        fuse_resp = {
            "status": "ok",
            "transform_4x4": [[1, 0, 0, 0]] * 4,
            "rmse": 0.03,
            "fitness": 0.9,
            "converged": True,
            "message": "",
        }
        with (
            patch("selfsuvis.pipeline.mapping.mapper._train_scene") as mock_train,
            patch("selfsuvis.pipeline.mapping.mapper._call_icp_fuse") as mock_fuse,
        ):
            mock_train.return_value = {
                "map_status": "success",
                "splat_path": "/maps/m1/splat.ply",
                "message": "done",
            }
            mock_fuse.return_value = fuse_resp
            result = run_mapper(
                "m1",
                _make_sfm_results(),
                target_splat_paths=["/maps/m0/splat.ply"],
            )
        assert len(result["icp_results"]) == 1
        icp = result["icp_results"][0]
        assert icp["source_splat"] == "/maps/m1/splat.ply"
        assert icp["target_splat"] == "/maps/m0/splat.ply"
        assert icp["converged"] is True
        assert abs(icp["rmse"] - 0.03) < 1e-6

    def test_icp_not_called_on_failed_scene(self):
        from selfsuvis.pipeline.mapping.mapper import run_mapper

        with (
            patch("selfsuvis.pipeline.mapping.mapper._train_scene") as mock_train,
            patch("selfsuvis.pipeline.mapping.mapper._call_icp_fuse") as mock_fuse,
        ):
            mock_train.return_value = {
                "map_status": "failed",
                "splat_path": None,
                "message": "error",
            }
            mock_fuse.return_value = None
            result = run_mapper(
                "m1",
                _make_sfm_results(),
                target_splat_paths=["/maps/m0/splat.ply"],
            )
        mock_fuse.assert_not_called()
        assert result["icp_results"] == []

    def test_icp_mapper_unavailable_does_not_fail(self):
        """ConnectionError from mapper service is a soft skip — run_mapper still succeeds."""
        from selfsuvis.pipeline.mapping.mapper import run_mapper

        with (
            patch("selfsuvis.pipeline.mapping.mapper._train_scene") as mock_train,
            patch("selfsuvis.pipeline.mapping.mapper._call_icp_fuse") as mock_fuse,
        ):
            mock_train.return_value = {
                "map_status": "success",
                "splat_path": "/maps/m1/splat.ply",
                "message": "done",
            }
            mock_fuse.return_value = None  # mapper unreachable → None
            result = run_mapper(
                "m1",
                _make_sfm_results(),
                target_splat_paths=["/maps/m0/splat.ply"],
            )
        assert result["map_status"] == "success"
        assert result["icp_results"] == []  # None entries are filtered out

    def test_multiple_targets_produce_multiple_icp_results(self):
        from selfsuvis.pipeline.mapping.mapper import run_mapper

        fuse_resp = {
            "status": "ok",
            "transform_4x4": [[1, 0, 0, 0]] * 4,
            "rmse": 0.02,
            "fitness": 0.95,
            "converged": True,
            "message": "",
        }
        with (
            patch("selfsuvis.pipeline.mapping.mapper._train_scene") as mock_train,
            patch("selfsuvis.pipeline.mapping.mapper._call_icp_fuse") as mock_fuse,
        ):
            mock_train.return_value = {
                "map_status": "success",
                "splat_path": "/maps/m1/splat.ply",
                "message": "done",
            }
            mock_fuse.return_value = fuse_resp
            result = run_mapper(
                "m1",
                _make_sfm_results(),
                target_splat_paths=["/maps/m0/splat.ply", "/maps/m_ref/splat.ply"],
            )
        assert len(result["icp_results"]) == 2

    def test_skipped_mission_has_empty_icp_results(self):
        from selfsuvis.pipeline.mapping.mapper import run_mapper

        # Only 5 frames — below MIN_FRAMES_FOR_3DGS
        result = run_mapper("m1", _make_sfm_results(5), target_splat_paths=["/maps/m0/splat.ply"])
        assert result["map_status"] == "skipped"
        assert result["icp_results"] == []

    def test_result_always_has_icp_results_key(self):
        """Callers should not need to .get("icp_results", []) — key is always present."""
        from selfsuvis.pipeline.mapping.mapper import run_mapper

        result = run_mapper("m1", _make_sfm_results(5))
        assert "icp_results" in result
