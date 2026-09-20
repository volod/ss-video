"""Unit tests for P2 global-map worker wiring.

Covers the three gaps fixed in worker/main.py _run_pass_a._db_and_map:

  1. update_mission_splat_path called after successful 3DGS so
     get_global_map_splats can return the splat to future missions.

  2. Bootstrap registration: first mission (no ICP targets) or non-converged
     ICP → register_mission called with identity transform so the chain
     does not stall forever.

  3. update_global_map_splat called when ICP produces a fused.ply.

The tests use a standalone _simulate_db_and_map coroutine that mirrors the
logic added to _db_and_map in worker/main.py exactly, driven against a real
MockConn so the tests verify DB state rather than just mock call counts.

Also tests:
  - update_mission_splat_path DB helper (pipeline/global_map_db.py)
  - The full discovery chain: update_mission_splat_path → register_mission
    → get_global_map_splats returns the splat for the next mission
  - Synthetic splat.ply creation via write_splat_from_arrays (no GPU).

No heavy ML imports (cv2, open_clip, torch) are required — worker.main is
not imported directly.
"""

import asyncio
import os
import tempfile
from typing import Any

import numpy as np

# ── synthetic splat helper ────────────────────────────────────────────────────


def _write_synthetic_splat(path: str, n: int = 100) -> str:
    """Write a minimal valid 3DGS splat.ply with n Gaussians.

    Uses random positions in a 10 m cube; identity quaternions; unit scales;
    zero SH.  Matches the 59-property format in pipeline/splat_io.py.
    """
    from selfsuvis.pipeline.mapping.splat_io import write_splat_from_arrays

    rng = np.random.RandomState(42)
    positions = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
    opacities = np.zeros(n, dtype=np.float32)
    scales = np.zeros((n, 3), dtype=np.float32)
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_splat_from_arrays(path, positions, opacities, scales, rotations)
    return path


# ── MockConn ──────────────────────────────────────────────────────────────────


class _Row(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class MockConn:
    """asyncpg Connection mock — in-memory tables for global_map, global_map_missions,
    and missions.  Tracks every execute() call for post-hoc assertion."""

    def __init__(self):
        self._global_maps: list = []
        self._gmm: list = []  # global_map_missions rows
        self._missions: list = []
        self._next_id = 1
        self.execute_calls: list = []  # (query_upper, args) for each call

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
        raise NotImplementedError(f"fetchval: {query[:60]}")

    async def fetch(self, query: str, *args) -> list:
        q = query.strip().upper()
        if "FROM GLOBAL_MAP ORDER" in q:
            return list(self._global_maps)
        if "FROM GLOBAL_MAP_MISSIONS" in q and "JOIN MISSIONS" in q:
            gm_id = args[0]
            result = []
            for gmm in self._gmm:
                if gmm["global_map_id"] == gm_id:
                    for m in self._missions:
                        if m["id"] == gmm["mission_id"] and m.get("splat_path"):
                            result.append(_Row(splat_path=m["splat_path"]))
            return result
        if "FROM GLOBAL_MAP_MISSIONS" in q:
            gm_id = args[0]
            return [_Row(**r) for r in self._gmm if r["global_map_id"] == gm_id]
        raise NotImplementedError(f"fetch: {query[:60]}")

    async def fetchrow(self, query: str, *args) -> _Row | None:
        q = query.strip().upper()
        if "FROM GLOBAL_MAP WHERE ID" in q:
            gm_id = args[0]
            for row in self._global_maps:
                if row["id"] == gm_id:
                    return row
            return None
        raise NotImplementedError(f"fetchrow: {query[:60]}")

    async def execute(self, query: str, *args) -> None:
        q = query.strip().upper()
        self.execute_calls.append((q, args))
        if "INSERT INTO GLOBAL_MAP_MISSIONS" in q:
            gm_id, mission_id, transform_json, reg_error, registered_at = args
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
        elif "UPDATE MISSIONS SET SPLAT_PATH" in q:
            splat_path, updated_at, mission_id = args
            for m in self._missions:
                if m["id"] == mission_id:
                    m["splat_path"] = splat_path
                    m["updated_at"] = updated_at
        else:
            raise NotImplementedError(f"execute: {query[:80]}")

    async def close(self):
        pass


# ── _db_and_map simulation ────────────────────────────────────────────────────
#
# This coroutine mirrors the logic added to _run_pass_a._db_and_map in
# worker/main.py.  Tests drive it directly so they can inspect MockConn
# state rather than patching call counts.  If the worker logic changes,
# this coroutine must be updated in sync.


async def _simulate_db_and_map(
    conn: MockConn,
    mission_id: str,
    global_map_id: int,
    mapper_result: dict[str, Any],
) -> None:
    """Execute the _db_and_map logic against a MockConn."""
    from selfsuvis.pipeline.storage.global_maps import (
        register_mission,
        update_global_map_splat,
        update_mission_splat_path,
    )

    primary_splat = mapper_result.get("splat_path")

    # Record splat path in missions table
    if primary_splat is not None:
        await update_mission_splat_path(conn, mission_id, primary_splat)

    # Persist ICP registrations; update global map fused splat when produced
    icp_registered = False
    for icp in mapper_result.get("icp_results", []):
        if icp.get("converged"):
            await register_mission(
                conn,
                global_map_id,
                mission_id,
                icp.get("transform_4x4"),
                icp.get("rmse"),
            )
            icp_registered = True
            if icp.get("fused_splat"):
                await update_global_map_splat(conn, global_map_id, icp["fused_splat"])

    # Bootstrap: register with identity when no ICP converged so the next
    # mission's get_global_map_splats call can find this splat as a target.
    if not icp_registered and primary_splat is not None:
        _identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        await register_mission(conn, global_map_id, mission_id, _identity, None)


def _run(coro):
    return asyncio.run(coro)


# ── mapper_result helpers ─────────────────────────────────────────────────────


def _mapper_success(splat: str, icp_results: list = None) -> dict:
    return {
        "map_status": "success",
        "splat_path": splat,
        "splat_paths": [splat],
        "scene_count": 1,
        "message": "done",
        "icp_results": icp_results or [],
    }


def _mapper_skipped() -> dict:
    return {
        "map_status": "skipped",
        "splat_path": None,
        "splat_paths": [],
        "scene_count": 0,
        "message": "not enough frames",
        "icp_results": [],
    }


def _icp_converged(src: str, tgt: str, fused: str, tx: float = 7.0) -> dict:
    return {
        "source_splat": src,
        "target_splat": tgt,
        "status": "ok",
        "converged": True,
        "transform_4x4": [[1, 0, 0, tx], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "rmse": 0.04,
        "fitness": 0.9,
        "message": "",
        "fused_splat": fused,
    }


def _icp_not_converged(src: str, tgt: str) -> dict:
    return {
        "source_splat": src,
        "target_splat": tgt,
        "status": "no_overlap",
        "converged": False,
        "transform_4x4": None,
        "rmse": None,
        "fitness": 0.0,
        "message": "insufficient overlap",
        "fused_splat": None,
    }


# ── update_mission_splat_path DB unit tests ───────────────────────────────────


class TestUpdateMissionSplatPath:
    def test_sets_splat_path_on_matching_mission(self):
        from selfsuvis.pipeline.storage.global_maps import update_mission_splat_path

        conn = MockConn()
        conn._missions = [{"id": "m1", "splat_path": None, "updated_at": 0.0}]
        _run(update_mission_splat_path(conn, "m1", "/maps/m1/splat.ply"))
        assert conn._missions[0]["splat_path"] == "/maps/m1/splat.ply"

    def test_execute_called_with_correct_args(self):
        from selfsuvis.pipeline.storage.global_maps import update_mission_splat_path

        conn = MockConn()
        conn._missions = [{"id": "m1", "splat_path": None, "updated_at": 0.0}]
        _run(update_mission_splat_path(conn, "m1", "/maps/m1/splat.ply"))
        assert len(conn.execute_calls) == 1
        query_upper, args = conn.execute_calls[0]
        assert "UPDATE MISSIONS SET SPLAT_PATH" in query_upper
        assert args[0] == "/maps/m1/splat.ply"
        assert args[2] == "m1"

    def test_updated_at_is_recent(self):
        from datetime import datetime, timezone

        from selfsuvis.pipeline.storage.global_maps import update_mission_splat_path

        conn = MockConn()
        conn._missions = [{"id": "m1", "splat_path": None, "updated_at": 0.0}]
        before = datetime.now(timezone.utc)
        _run(update_mission_splat_path(conn, "m1", "/maps/m1/splat.ply"))
        after = datetime.now(timezone.utc)
        _, args = conn.execute_calls[0]
        assert before <= args[1] <= after


# ── discovery chain tests ─────────────────────────────────────────────────────


class TestGetGlobalMapSplatsAfterUpdate:
    """update_mission_splat_path → register_mission → get_global_map_splats returns it."""

    def test_splat_discoverable_after_mission_update(self):
        from selfsuvis.pipeline.storage.global_maps import (
            get_global_map_splats,
            register_mission,
            update_mission_splat_path,
        )

        conn = MockConn()
        conn._global_maps.append(_Row(id=1, origin_lat=48.0, origin_lon=11.0))
        conn._missions.append({"id": "m1", "splat_path": None, "updated_at": 0.0})

        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        _run(register_mission(conn, 1, "m1", identity, None))
        _run(update_mission_splat_path(conn, "m1", "/maps/m1/splat.ply"))

        splats = _run(get_global_map_splats(conn, 1))
        assert splats == ["/maps/m1/splat.ply"]

    def test_splat_not_visible_before_update(self):
        from selfsuvis.pipeline.storage.global_maps import get_global_map_splats, register_mission

        conn = MockConn()
        conn._global_maps.append(_Row(id=1, origin_lat=48.0, origin_lon=11.0))
        conn._missions.append({"id": "m1", "splat_path": None, "updated_at": 0.0})
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        _run(register_mission(conn, 1, "m1", identity, None))
        # splat_path still None
        splats = _run(get_global_map_splats(conn, 1))
        assert splats == []

    def test_two_missions_both_discoverable(self):
        from selfsuvis.pipeline.storage.global_maps import (
            get_global_map_splats,
            register_mission,
            update_mission_splat_path,
        )

        conn = MockConn()
        conn._global_maps.append(_Row(id=1, origin_lat=48.0, origin_lon=11.0))
        conn._missions = [
            {"id": "m1", "splat_path": None, "updated_at": 0.0},
            {"id": "m2", "splat_path": None, "updated_at": 0.0},
        ]
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        _run(register_mission(conn, 1, "m1", identity, None))
        _run(update_mission_splat_path(conn, "m1", "/maps/m1/splat.ply"))
        _run(register_mission(conn, 1, "m2", identity, None))
        _run(update_mission_splat_path(conn, "m2", "/maps/m2/splat.ply"))

        splats = _run(get_global_map_splats(conn, 1))
        assert set(splats) == {"/maps/m1/splat.ply", "/maps/m2/splat.ply"}


# ── _db_and_map logic tests ───────────────────────────────────────────────────


class TestDbAndMapFirstMission:
    """First mission at a site: no ICP targets, 3DGS succeeds."""

    def _setup(self, splat="/maps/m1/splat.ply"):
        conn = MockConn()
        conn._global_maps.append(
            _Row(id=1, origin_lat=48.0, origin_lon=11.0, splat_path=None, updated_at=0.0)
        )
        conn._missions.append({"id": "m1", "splat_path": None, "updated_at": 0.0})
        _run(_simulate_db_and_map(conn, "m1", 1, _mapper_success(splat)))
        return conn

    def test_missions_splat_path_set(self):
        conn = self._setup()
        assert conn._missions[0]["splat_path"] == "/maps/m1/splat.ply"

    def test_bootstrap_registration_inserted(self):
        conn = self._setup()
        assert len(conn._gmm) == 1
        row = conn._gmm[0]
        assert row["mission_id"] == "m1"
        assert row["global_map_id"] == 1

    def test_bootstrap_registration_uses_identity_transform(self):
        import json

        conn = self._setup()
        transform = json.loads(conn._gmm[0]["registration_transform_json"])
        assert transform == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def test_bootstrap_registration_error_is_none(self):
        conn = self._setup()
        assert conn._gmm[0]["registration_error"] is None

    def test_global_map_splat_path_not_updated(self):
        """No ICP fusion → global_map.splat_path stays None."""
        conn = self._setup()
        assert conn._global_maps[0]["splat_path"] is None

    def test_splat_discoverable_after_first_mission(self):
        from selfsuvis.pipeline.storage.global_maps import get_global_map_splats

        conn = self._setup()
        splats = _run(get_global_map_splats(conn, 1))
        assert splats == ["/maps/m1/splat.ply"]


class TestDbAndMapSecondMissionIcpConverged:
    """Second mission: ICP converges and produces a fused.ply."""

    def _setup(self):
        conn = MockConn()
        conn._global_maps.append(
            _Row(id=1, origin_lat=48.0, origin_lon=11.0, splat_path=None, updated_at=0.0)
        )
        conn._missions.append({"id": "m2", "splat_path": None, "updated_at": 0.0})
        result = _mapper_success(
            "/maps/m2/splat.ply",
            icp_results=[
                _icp_converged("/maps/m2/splat.ply", "/maps/m1/splat.ply", "/maps/m2/fused.ply")
            ],
        )
        _run(_simulate_db_and_map(conn, "m2", 1, result))
        return conn

    def test_missions_splat_path_set(self):
        conn = self._setup()
        assert conn._missions[0]["splat_path"] == "/maps/m2/splat.ply"

    def test_register_mission_called_with_icp_transform(self):
        import json

        conn = self._setup()
        assert len(conn._gmm) == 1
        transform = json.loads(conn._gmm[0]["registration_transform_json"])
        assert transform == [[1, 0, 0, 7], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def test_register_mission_has_icp_rmse(self):
        conn = self._setup()
        assert abs(conn._gmm[0]["registration_error"] - 0.04) < 1e-6

    def test_global_map_splat_updated_to_fused(self):
        conn = self._setup()
        assert conn._global_maps[0]["splat_path"] == "/maps/m2/fused.ply"

    def test_no_extra_bootstrap_registration(self):
        """Only one registration row — identity bootstrap must not fire."""
        conn = self._setup()
        assert len(conn._gmm) == 1

    def test_icp_transform_is_not_identity(self):
        import json

        conn = self._setup()
        transform = json.loads(conn._gmm[0]["registration_transform_json"])
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        assert transform != identity


class TestDbAndMapIcpNotConverged:
    """ICP target existed but did not converge: bootstrap registration fires."""

    def _setup(self):
        conn = MockConn()
        conn._global_maps.append(
            _Row(id=1, origin_lat=48.0, origin_lon=11.0, splat_path=None, updated_at=0.0)
        )
        conn._missions.append({"id": "m2", "splat_path": None, "updated_at": 0.0})
        result = _mapper_success(
            "/maps/m2/splat.ply",
            icp_results=[_icp_not_converged("/maps/m2/splat.ply", "/maps/m1/splat.ply")],
        )
        _run(_simulate_db_and_map(conn, "m2", 1, result))
        return conn

    def test_missions_splat_path_still_set(self):
        conn = self._setup()
        assert conn._missions[0]["splat_path"] == "/maps/m2/splat.ply"

    def test_bootstrap_registration_fires(self):
        import json

        conn = self._setup()
        assert len(conn._gmm) == 1
        transform = json.loads(conn._gmm[0]["registration_transform_json"])
        assert transform == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def test_global_map_splat_not_updated(self):
        conn = self._setup()
        assert conn._global_maps[0]["splat_path"] is None

    def test_splat_still_discoverable_for_next_mission(self):
        from selfsuvis.pipeline.storage.global_maps import get_global_map_splats

        conn = self._setup()
        splats = _run(get_global_map_splats(conn, 1))
        assert splats == ["/maps/m2/splat.ply"]


class TestDbAndMapMapperSkipped:
    """Mapper returned skipped (too few frames): no DB writes."""

    def _setup(self):
        conn = MockConn()
        conn._global_maps.append(_Row(id=1, origin_lat=48.0, origin_lon=11.0))
        conn._missions.append({"id": "m1", "splat_path": None, "updated_at": 0.0})
        _run(_simulate_db_and_map(conn, "m1", 1, _mapper_skipped()))
        return conn

    def test_no_splat_path_set(self):
        conn = self._setup()
        assert conn._missions[0]["splat_path"] is None

    def test_no_registration_inserted(self):
        conn = self._setup()
        assert conn._gmm == []

    def test_no_execute_calls(self):
        conn = self._setup()
        assert conn.execute_calls == []

    def test_splat_not_discoverable(self):
        from selfsuvis.pipeline.storage.global_maps import get_global_map_splats

        conn = self._setup()
        splats = _run(get_global_map_splats(conn, 1))
        assert splats == []


class TestDbAndMapMultipleIcpTargets:
    """Multiple targets: each converged ICP fires a separate update_global_map_splat."""

    def test_last_fused_splat_wins(self):
        conn = MockConn()
        conn._global_maps.append(
            _Row(id=1, origin_lat=48.0, origin_lon=11.0, splat_path=None, updated_at=0.0)
        )
        conn._missions.append({"id": "m3", "splat_path": None, "updated_at": 0.0})

        result = _mapper_success(
            "/maps/m3/splat.ply",
            icp_results=[
                _icp_converged(
                    "/maps/m3/splat.ply", "/maps/m1/splat.ply", "/maps/m3/fused_vs_m1.ply", tx=3.0
                ),
                _icp_converged(
                    "/maps/m3/splat.ply", "/maps/m2/splat.ply", "/maps/m3/fused_vs_m2.ply", tx=5.0
                ),
            ],
        )
        _run(_simulate_db_and_map(conn, "m3", 1, result))

        # Only one gmm row (UPSERT on conflict)
        assert len(conn._gmm) == 1
        # global_map.splat_path reflects the last fused splat written
        assert conn._global_maps[0]["splat_path"] == "/maps/m3/fused_vs_m2.ply"

    def test_no_bootstrap_when_any_icp_converged(self):
        import json

        conn = MockConn()
        conn._global_maps.append(
            _Row(id=1, origin_lat=48.0, origin_lon=11.0, splat_path=None, updated_at=0.0)
        )
        conn._missions.append({"id": "m3", "splat_path": None, "updated_at": 0.0})
        result = _mapper_success(
            "/maps/m3/splat.ply",
            icp_results=[
                _icp_converged("/maps/m3/splat.ply", "/maps/m1/splat.ply", "/maps/m3/fused.ply"),
            ],
        )
        _run(_simulate_db_and_map(conn, "m3", 1, result))
        # Single row, non-identity transform
        assert len(conn._gmm) == 1
        transform = json.loads(conn._gmm[0]["registration_transform_json"])
        assert transform != [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class TestDbAndMapFusedSplatNone:
    """ICP converged but fused_splat is None (fusion failed): only register, no map update."""

    def test_register_called_but_global_map_not_updated(self):
        conn = MockConn()
        conn._global_maps.append(
            _Row(id=1, origin_lat=48.0, origin_lon=11.0, splat_path=None, updated_at=0.0)
        )
        conn._missions.append({"id": "m2", "splat_path": None, "updated_at": 0.0})
        icp = _icp_converged("/maps/m2/splat.ply", "/maps/m1/splat.ply", None)
        icp["fused_splat"] = None  # fusion step failed
        result = _mapper_success("/maps/m2/splat.ply", icp_results=[icp])
        _run(_simulate_db_and_map(conn, "m2", 1, result))

        # Registration happened (ICP converged)
        assert len(conn._gmm) == 1
        # But global_map.splat_path not updated (no fused file)
        assert conn._global_maps[0]["splat_path"] is None


# ── synthetic splat.ply tests ─────────────────────────────────────────────────


class TestSyntheticSplatFiles:
    """Verify the synthetic splat helper produces valid 3DGS PLY files."""

    def test_write_and_read_roundtrip(self):
        from selfsuvis.pipeline.mapping.splat_io import read_splat, splat_count

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_synthetic_splat(os.path.join(tmp, "test.ply"), n=200)
            assert splat_count(path) == 200
            data = read_splat(path)
            assert len(data) == 200

    def test_all_59_properties_present(self):
        from selfsuvis.pipeline.mapping.splat_io import ALL_PROPERTIES, read_splat

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_synthetic_splat(os.path.join(tmp, "test.ply"), n=50)
            data = read_splat(path)
            for prop in ALL_PROPERTIES:
                assert prop in data.dtype.names, f"Missing: {prop}"

    def test_rotations_are_unit_quaternions(self):
        from selfsuvis.pipeline.mapping.splat_io import read_splat

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_synthetic_splat(os.path.join(tmp, "test.ply"), n=50)
            data = read_splat(path)
            quats = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]])
            norms = np.linalg.norm(quats, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_merge_produces_correct_count(self):
        from selfsuvis.pipeline.mapping.splat_io import merge_splats, splat_count

        with tempfile.TemporaryDirectory() as tmp:
            a = _write_synthetic_splat(os.path.join(tmp, "a.ply"), n=100)
            b = _write_synthetic_splat(os.path.join(tmp, "b.ply"), n=150)
            total = merge_splats([a, b], os.path.join(tmp, "fused.ply"))
            assert total == 250
            assert splat_count(os.path.join(tmp, "fused.ply")) == 250

    def test_is_splat_ply_recognises_synthetic(self):
        from selfsuvis.pipeline.mapping.splat_io import is_splat_ply

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_synthetic_splat(os.path.join(tmp, "test.ply"), n=10)
            assert is_splat_ply(path) is True

    def test_typical_small_mission_size(self):
        """Confirm we can create a small-mission-sized splat (50K Gaussians)."""
        from selfsuvis.pipeline.mapping.splat_io import splat_count

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_synthetic_splat(os.path.join(tmp, "medium.ply"), n=50_000)
            assert splat_count(path) == 50_000
