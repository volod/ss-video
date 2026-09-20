"""Unit tests for worker._run_pass_a — Phase 2 ICP wire-up.

All external dependencies (pycolmap, asyncpg, nerfstudio, postgres) are mocked.
Synthetic SfM frame data drives every test path without real video or GPU.

Because asyncpg / pipeline.sfm / pipeline.mapping.mapper are optional and may not be
installed in the unit-test environment, we inject fake stub modules into
sys.modules before any test runs so that unittest.mock.patch can resolve them.
worker.handlers.postflight imports pipeline.storage (stubbed below).
"""

import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Inject stub modules for optional deps that may not be installed
# ---------------------------------------------------------------------------


def _make_stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _ensure_stub(name: str, **attrs) -> types.ModuleType:
    """Only inject if not already present (don't overwrite real modules)."""
    if name not in sys.modules:
        return _make_stub(name, **attrs)
    return sys.modules[name]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stubs for worker.main transitive imports (avoid cv2 / job-db / etc.)
# Must be injected BEFORE importing worker.main.
# ---------------------------------------------------------------------------

# Stub pipeline.workflows to avoid heavy model imports (cv2 / torch / etc.)
_workflows_stub = _make_stub("selfsuvis.pipeline.workflows")
_workflows_stub.VideoIndexer = MagicMock()  # type: ignore[attr-defined]

# Stub pipeline.storage.processed entrypoints used by worker.main.
_make_stub(
    "selfsuvis.pipeline.storage.processed",
    ainit_db=MagicMock(),
    aget_by_hash=MagicMock(),
    aupsert=MagicMock(),
    get_by_hash=MagicMock(),
)

# ---------------------------------------------------------------------------
# Stubs for optional pipeline deps (not installed in unit-test venv)
# ---------------------------------------------------------------------------

# asyncpg stub — use _ensure_stub so all test files share the same module
# object. _make_stub would create a new object, breaking patches in files
# (e.g. test_gpu_isolation.py) that captured sys.modules["asyncpg"] earlier.
# Pool must be present so that db.py type annotations (asyncpg.Pool) resolve
# when this stub replaces the real module before those files are imported.
_asyncpg_stub = _ensure_stub("asyncpg", connect=AsyncMock(), Pool=MagicMock())

# pipeline.mapping.sfm stub
_sfm_stub = _make_stub("selfsuvis.pipeline.mapping.sfm", run_sfm=MagicMock())

# pipeline.mapper is NOT stubbed at module level — worker.main imports it
# lazily inside _run_pass_a (not at the module top level), so no stub is
# needed here.  Individual tests patch pipeline.mapper.run_mapper as needed.

# pipeline.mapping.gps_registration stub
_ensure_stub(
    "selfsuvis.pipeline.mapping.gps_registration",
    register_mission_gps=MagicMock(),
    gps_to_enu=MagicMock(),
)

# NOTE: Do not stub selfsuvis.pipeline.storage.global_maps. It is dependency-light
# and other tests rely on importing the real module.


# ---------------------------------------------------------------------------
# Synthetic test fixtures
# ---------------------------------------------------------------------------

_SFM_FRAME = {
    "frame_path": "/tmp/f.jpg",
    "t_sec": 1.0,
    "pose_status": "success",
    "scene_index": 0,
    "pose_json": "{}",
}
_SFM_OUT = {"frames": [_SFM_FRAME] * 35, "scene_count": 1}
_ENU_ORIGIN = (48.0, 11.0, 100.0)  # (lat, lon, alt)
_GLOBAL_POSES = [{"tx": 0.0, "ty": 0.0, "tz": 0.0}] * 35
_IDENTITY_T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
_ICP_CONVERGED = {
    "converged": True,
    "transform_4x4": _IDENTITY_T,
    "rmse": 0.05,
    "fitness": 0.9,
}
_ICP_DIVERGED = {
    "converged": False,
    "transform_4x4": None,
    "rmse": None,
    "fitness": 0.0,
}
_MAPPER_RESULT = {
    "map_status": "success",
    "splat_paths": ["/tmp/splat.ply"],
    "icp_results": [_ICP_CONVERGED],
}
_GLOBAL_MAP_ID = 42
_SPLAT_PATHS = ["/tmp/global.ply"]

VIDEO_PATH = "/tmp/video.mp4"
VIDEO_ID = "vid_001"
MISSION_ID = "mission_001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger():
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


def _fake_asyncpg_conn():
    conn = AsyncMock()
    conn.close = AsyncMock()
    return conn


def _get_run_pass_a():
    import selfsuvis.worker.handlers.postflight as ph

    return ph._run_pass_a


# ---------------------------------------------------------------------------
# Patch context manager — resets asyncpg.connect to a fresh AsyncMock per test
# so state doesn't bleed between tests.
# ---------------------------------------------------------------------------


def _all_patches(conn, mapper_result=None, enu_reg=None):
    """Return a list of context managers for the full happy-path mock set."""
    if mapper_result is None:
        mapper_result = _MAPPER_RESULT
    if enu_reg is None:
        enu_reg = (_ENU_ORIGIN, _GLOBAL_POSES)

    return [
        patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
        patch(
            "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
            return_value=enu_reg,
        ),
        patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
        patch("selfsuvis.pipeline.mapping.mapper.run_mapper", return_value=mapper_result),
        patch(
            "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
            new_callable=AsyncMock,
            return_value=_GLOBAL_MAP_ID,
        ),
        patch(
            "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
            new_callable=AsyncMock,
            return_value=_SPLAT_PATHS,
        ),
        patch("selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock),
    ]


class _MultiPatch:
    """Apply a list of patch context managers together."""

    def __init__(self, patches):
        self._patches = patches
        self._mocks = []

    def __enter__(self):
        self._mocks = [p.__enter__() for p in self._patches]
        return self._mocks

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------


class TestPassAHappyPath(unittest.TestCase):
    def setUp(self):
        self.conn = _fake_asyncpg_conn()

    def _run(self, mapper_result=None, enu_reg=None, splat_paths=None, extra_patches=None):
        conn = self.conn
        if mapper_result is None:
            mapper_result = _MAPPER_RESULT
        if enu_reg is None:
            enu_reg = (_ENU_ORIGIN, _GLOBAL_POSES)
        if splat_paths is None:
            splat_paths = _SPLAT_PATHS

        patches = [
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                return_value=enu_reg,
            ),
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("selfsuvis.pipeline.mapping.mapper.run_mapper", return_value=mapper_result),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
                return_value=_GLOBAL_MAP_ID,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
                return_value=splat_paths,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock
            ),
        ]
        if extra_patches:
            patches.extend(extra_patches)

        cms = []
        for p in patches:
            cm = p.__enter__()
            cms.append((p, cm))

        # Build named mocks from the patched targets
        names = [
            "mock_sfm",
            "mock_gps_reg",
            "mock_connect",
            "mock_mapper",
            "mock_get_map",
            "mock_get_splats",
            "mock_reg_mission",
        ]
        named = {}
        for i, (name, (p, cm)) in enumerate(zip(names, cms)):
            named[name] = cm

        self._logger = _make_logger()
        _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, self._logger)

        for p, _ in reversed(cms):
            p.__exit__(None, None, None)

        return named

    def test_sfm_called_with_correct_args(self):
        m = self._run()
        m["mock_sfm"].assert_called_once_with(VIDEO_PATH, VIDEO_ID, MISSION_ID)

    def test_gps_registration_called_with_sfm_frames(self):
        m = self._run()
        m["mock_gps_reg"].assert_called_once_with(_SFM_OUT["frames"])

    def test_get_or_create_global_map_called_with_enu_origin(self):
        m = self._run()
        m["mock_get_map"].assert_called_once()
        args = m["mock_get_map"].call_args.args
        lat, lon, alt = _ENU_ORIGIN
        self.assertEqual(args[1], lat)
        self.assertEqual(args[2], lon)
        self.assertEqual(args[3], alt)

    def test_get_global_map_splats_called_with_global_map_id(self):
        m = self._run()
        m["mock_get_splats"].assert_called_once()
        self.assertEqual(m["mock_get_splats"].call_args.args[1], _GLOBAL_MAP_ID)

    def test_mapper_called_with_splat_paths(self):
        m = self._run()
        m["mock_mapper"].assert_called_once()
        kwargs = m["mock_mapper"].call_args.kwargs
        self.assertEqual(kwargs.get("target_splat_paths"), _SPLAT_PATHS)

    def test_register_mission_called_for_converged_icp(self):
        m = self._run()
        m["mock_reg_mission"].assert_called_once()
        args = m["mock_reg_mission"].call_args.args
        self.assertEqual(args[1], _GLOBAL_MAP_ID)
        self.assertEqual(args[2], MISSION_ID)
        self.assertEqual(args[3], _IDENTITY_T)
        self.assertAlmostEqual(args[4], 0.05, places=6)

    def test_conn_closed_after_success(self):
        self._run()
        self.conn.close.assert_awaited_once()

    def test_no_warning_on_happy_path(self):
        self._run()
        self._logger.warning.assert_not_called()

    def test_sfm_done_info_logged(self):
        self._run()
        info_calls = [c[0][0] for c in self._logger.info.call_args_list]
        self.assertTrue(any("SfM done" in msg for msg in info_calls))

    def test_mapper_done_info_logged(self):
        self._run()
        info_calls = [c[0][0] for c in self._logger.info.call_args_list]
        self.assertTrue(any("mapper done" in msg for msg in info_calls))


# ---------------------------------------------------------------------------
# SfM execution failures
# ---------------------------------------------------------------------------


class TestPassASfMFailures(unittest.TestCase):
    def test_sfm_runtime_exception_returns_early(self):
        with patch(
            "selfsuvis.pipeline.mapping.sfm.run_sfm",
            side_effect=RuntimeError("colmap crash"),
        ):
            with patch("selfsuvis.pipeline.mapping.mapper.run_mapper") as mock_mapper:
                logger = _make_logger()
                _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
                mock_mapper.assert_not_called()
                logger.warning.assert_called()
                self.assertIn("SfM failed", logger.warning.call_args[0][0])

    def test_sfm_value_error_returns_early(self):
        with patch("selfsuvis.pipeline.mapping.sfm.run_sfm", side_effect=ValueError("bad video")):
            logger = _make_logger()
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
            logger.warning.assert_called()

    def test_sfm_exception_includes_mission_id_in_log(self):
        with patch("selfsuvis.pipeline.mapping.sfm.run_sfm", side_effect=RuntimeError("crash")):
            logger = _make_logger()
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
            # Second format arg should be the mission_id
            warn_args = logger.warning.call_args[0]
            self.assertEqual(warn_args[1], MISSION_ID)


# ---------------------------------------------------------------------------
# GPS registration failures
# ---------------------------------------------------------------------------


class TestPassAGpsRegFailures(unittest.TestCase):
    def _run_with_gps_exc(self, exc):
        conn = _fake_asyncpg_conn()
        with (
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                side_effect=exc,
            ),
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch(
                "selfsuvis.pipeline.mapping.mapper.run_mapper",
                return_value={"map_status": "success", "splat_paths": [], "icp_results": []},
            ) as mock_mapper,
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
            ) as mock_get_map,
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock
            ),
        ):
            logger = _make_logger()
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
            return logger, mock_get_map, mock_mapper

    def test_gps_reg_exception_no_global_map_lookup(self):
        _, mock_get_map, _ = self._run_with_gps_exc(Exception("no GPS"))
        mock_get_map.assert_not_called()

    def test_gps_reg_exception_mapper_still_called(self):
        _, _, mock_mapper = self._run_with_gps_exc(Exception("no GPS"))
        mock_mapper.assert_called_once()

    def test_gps_reg_exception_mapper_called_with_empty_targets(self):
        _, _, mock_mapper = self._run_with_gps_exc(ValueError("oops"))
        kwargs = mock_mapper.call_args.kwargs
        self.assertEqual(kwargs.get("target_splat_paths"), [])

    def test_gps_reg_failure_logs_warning_with_mission_id(self):
        logger, _, _ = self._run_with_gps_exc(RuntimeError("no gps"))
        logger.warning.assert_called()
        self.assertIn(MISSION_ID, logger.warning.call_args[0][1])

    def test_gps_reg_failure_does_not_propagate(self):
        # Should not raise
        self._run_with_gps_exc(RuntimeError("fatal gps error"))


# ---------------------------------------------------------------------------
# asyncpg / mapper dep import failures (hide module from sys.modules)
# ---------------------------------------------------------------------------


class TestPassAMapperImportFailures(unittest.TestCase):
    def test_asyncpg_connect_failure_returns_gracefully(self):
        """When asyncpg.connect raises, the mapper/DB step is skipped gracefully.

        asyncpg is now a top-level import in worker.main (not lazy), so simulating
        a missing module via sys.modules is not applicable.  Instead we simulate
        an unreachable database by raising on connect — the real production failure
        mode for environments without a live PostgreSQL instance.
        """
        with (
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                return_value=(_ENU_ORIGIN, _GLOBAL_POSES),
            ),
            patch(
                "asyncpg.connect", new_callable=AsyncMock, side_effect=OSError("connection refused")
            ),
            patch("selfsuvis.pipeline.mapping.mapper.run_mapper", return_value=_MAPPER_RESULT),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock
            ),
        ):
            logger = _make_logger()
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
            # mapper/DB step failure is logged as a warning
            logger.warning.assert_called()

    def test_mapper_hidden_returns_gracefully(self):
        real_mapper = sys.modules.get("selfsuvis.pipeline.mapping.mapper")
        sys.modules["selfsuvis.pipeline.mapping.mapper"] = None  # type: ignore[assignment]
        try:
            with (
                patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
                patch(
                    "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                    return_value=(_ENU_ORIGIN, _GLOBAL_POSES),
                ),
            ):
                logger = _make_logger()
                _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
                # Must not raise
        finally:
            sys.modules["selfsuvis.pipeline.mapping.mapper"] = real_mapper  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# asyncpg connection failures
# ---------------------------------------------------------------------------


class TestPassAConnectionFailures(unittest.TestCase):
    def test_asyncpg_connect_exception_does_not_propagate(self):
        with (
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                return_value=(_ENU_ORIGIN, _GLOBAL_POSES),
            ),
            patch("asyncpg.connect", side_effect=ConnectionRefusedError("postgres down")),
            patch("selfsuvis.pipeline.mapping.mapper.run_mapper", return_value=_MAPPER_RESULT),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock
            ),
        ):
            logger = _make_logger()
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
            logger.warning.assert_called()

    def test_conn_closed_even_if_mapper_raises(self):
        conn = _fake_asyncpg_conn()
        with (
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                return_value=(_ENU_ORIGIN, _GLOBAL_POSES),
            ),
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch(
                "selfsuvis.pipeline.mapping.mapper.run_mapper",
                side_effect=RuntimeError("nerfstudio OOM"),
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
                return_value=_GLOBAL_MAP_ID,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
                return_value=_SPLAT_PATHS,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock
            ),
        ):
            logger = _make_logger()
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
            conn.close.assert_awaited_once()

    def test_conn_closed_even_if_register_mission_raises(self):
        conn = _fake_asyncpg_conn()
        with (
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                return_value=(_ENU_ORIGIN, _GLOBAL_POSES),
            ),
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("selfsuvis.pipeline.mapping.mapper.run_mapper", return_value=_MAPPER_RESULT),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
                return_value=_GLOBAL_MAP_ID,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
                return_value=_SPLAT_PATHS,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission",
                new_callable=AsyncMock,
                side_effect=RuntimeError("DB write fail"),
            ),
        ):
            logger = _make_logger()
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
            conn.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# ICP result filtering
# ---------------------------------------------------------------------------


class TestPassAIcpFiltering(unittest.TestCase):
    def _run_with_icp(self, icp_results):
        conn = _fake_asyncpg_conn()
        mapper_result = {
            "map_status": "success",
            "splat_paths": ["/tmp/splat.ply"],
            "icp_results": icp_results,
        }
        with (
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                return_value=(_ENU_ORIGIN, _GLOBAL_POSES),
            ),
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("selfsuvis.pipeline.mapping.mapper.run_mapper", return_value=mapper_result),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
                return_value=_GLOBAL_MAP_ID,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
                return_value=_SPLAT_PATHS,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock
            ) as mock_reg,
        ):
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, _make_logger())
            return mock_reg

    def test_no_icp_results_register_mission_not_called(self):
        mock_reg = self._run_with_icp([])
        mock_reg.assert_not_called()

    def test_diverged_icp_register_mission_not_called(self):
        mock_reg = self._run_with_icp([_ICP_DIVERGED])
        mock_reg.assert_not_called()

    def test_converged_icp_register_mission_called_once(self):
        mock_reg = self._run_with_icp([_ICP_CONVERGED])
        mock_reg.assert_called_once()

    def test_mixed_icp_only_converged_registered(self):
        """2 converged + 2 diverged → register_mission called exactly 2 times."""
        icp_results = [_ICP_CONVERGED, _ICP_DIVERGED, _ICP_CONVERGED, _ICP_DIVERGED]
        mock_reg = self._run_with_icp(icp_results)
        self.assertEqual(mock_reg.call_count, 2)

    def test_all_diverged_no_registration(self):
        mock_reg = self._run_with_icp([_ICP_DIVERGED, _ICP_DIVERGED, _ICP_DIVERGED])
        mock_reg.assert_not_called()

    def test_converged_icp_passes_correct_transform(self):
        custom_T = [[2, 0, 0, 1], [0, 2, 0, 2], [0, 0, 2, 3], [0, 0, 0, 1]]
        icp = {"converged": True, "transform_4x4": custom_T, "rmse": 0.01, "fitness": 0.99}
        mock_reg = self._run_with_icp([icp])
        args = mock_reg.call_args.args
        self.assertEqual(args[3], custom_T)
        self.assertAlmostEqual(args[4], 0.01, places=6)

    def test_icp_missing_converged_key_treated_as_not_converged(self):
        """ICP entry with no 'converged' key → falsy → not registered."""
        icp = {"transform_4x4": _IDENTITY_T, "rmse": 0.05}
        mock_reg = self._run_with_icp([icp])
        mock_reg.assert_not_called()

    def test_converged_icp_passes_global_map_id_and_mission_id(self):
        mock_reg = self._run_with_icp([_ICP_CONVERGED])
        args = mock_reg.call_args.args
        self.assertEqual(args[1], _GLOBAL_MAP_ID)
        self.assertEqual(args[2], MISSION_ID)


# ---------------------------------------------------------------------------
# Scene count propagation
# ---------------------------------------------------------------------------


class TestPassASceneCount(unittest.TestCase):
    def _run_with_sfm_out(self, sfm_out):
        conn = _fake_asyncpg_conn()
        with (
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=sfm_out),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                return_value=(_ENU_ORIGIN, _GLOBAL_POSES),
            ),
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch(
                "selfsuvis.pipeline.mapping.mapper.run_mapper",
                return_value={"map_status": "success", "splat_paths": [], "icp_results": []},
            ) as mock_mapper,
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
                return_value=_GLOBAL_MAP_ID,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock
            ),
        ):
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, _make_logger())
            return mock_mapper

    def test_scene_count_passed_to_mapper(self):
        sfm_out = {"frames": [_SFM_FRAME] * 20, "scene_count": 3}
        mock_mapper = self._run_with_sfm_out(sfm_out)
        kwargs = mock_mapper.call_args.kwargs
        self.assertEqual(kwargs.get("scene_count"), 3)

    def test_default_scene_count_is_1(self):
        """SfM output without scene_count key defaults to 1."""
        sfm_out = {"frames": [_SFM_FRAME]}  # no scene_count key
        mock_mapper = self._run_with_sfm_out(sfm_out)
        kwargs = mock_mapper.call_args.kwargs
        self.assertEqual(kwargs.get("scene_count"), 1)

    def test_scene_count_zero_passed_as_is(self):
        sfm_out = {"frames": [], "scene_count": 0}
        mock_mapper = self._run_with_sfm_out(sfm_out)
        kwargs = mock_mapper.call_args.kwargs
        self.assertEqual(kwargs.get("scene_count"), 0)


# ---------------------------------------------------------------------------
# GPS registration success: logged correctly
# ---------------------------------------------------------------------------


class TestPassAGpsRegistrationSuccess(unittest.TestCase):
    def test_gps_registration_done_logged(self):
        conn = _fake_asyncpg_conn()
        with (
            patch("selfsuvis.pipeline.mapping.sfm.run_sfm", return_value=_SFM_OUT),
            patch(
                "selfsuvis.pipeline.mapping.gps_registration.register_mission_gps",
                return_value=(_ENU_ORIGIN, _GLOBAL_POSES),
            ),
            patch("asyncpg.connect", new_callable=AsyncMock, return_value=conn),
            patch("selfsuvis.pipeline.mapping.mapper.run_mapper", return_value=_MAPPER_RESULT),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_or_create_global_map",
                new_callable=AsyncMock,
                return_value=_GLOBAL_MAP_ID,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.get_global_map_splats",
                new_callable=AsyncMock,
                return_value=_SPLAT_PATHS,
            ),
            patch(
                "selfsuvis.pipeline.storage.global_maps.register_mission", new_callable=AsyncMock
            ),
        ):
            logger = _make_logger()
            _get_run_pass_a()(VIDEO_PATH, VIDEO_ID, MISSION_ID, {}, logger)
            info_calls = [c[0][0] for c in logger.info.call_args_list]
            self.assertTrue(any("GPS registration done" in msg for msg in info_calls))


if __name__ == "__main__":
    unittest.main()
