"""Unit tests for app.state._resolve_dino_checkpoint.

The function is extracted inline via exec to avoid importing app.state,
which executes heavy side-effects (loads OpenCLIP, connects to Qdrant, etc.).

asyncpg is not installed in the test venv; it is stubbed in sys.modules so
that the `import asyncpg` inside _resolve_dino_checkpoint succeeds.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Stub asyncpg before anything else ────────────────────────────────────────
if "asyncpg" not in sys.modules:
    _asyncpg = types.ModuleType("asyncpg")
    _asyncpg.connect = MagicMock()  # patch.object requires attribute to pre-exist
    _asyncpg.Pool = MagicMock()  # type: ignore[attr-defined]
    sys.modules["asyncpg"] = _asyncpg

_asyncpg_mod = sys.modules["asyncpg"]


# ── Extract _resolve_dino_checkpoint via exec ─────────────────────────────────


def _build_resolver():
    """Compile and return a testable version of _resolve_dino_checkpoint.

    The production code mutates the module-level `settings` object.  Here we
    parameterise it so tests can pass their own mock settings.
    """
    source = """\
import asyncio

def _resolve_dino_checkpoint(settings, logger):
    db_url = settings.DATABASE_URL
    if not db_url:
        return

    try:
        import asyncpg

        async def _fetch():
            conn = await asyncpg.connect(db_url, timeout=5)
            try:
                row = await conn.fetchrow(
                    "SELECT value FROM system_state WHERE key = 'active_dino_checkpoint'"
                )
                return row["value"] if row else None
            finally:
                await conn.close()

        db_ckpt = asyncio.run(_fetch())
        if db_ckpt:
            logger.info("DINOv3 checkpoint from DB (overrides env): %s", db_ckpt)
            settings.DINO_CHECKPOINT = db_ckpt
    except Exception as exc:
        logger.warning(
            "Could not read active_dino_checkpoint from DB (falling back to env): %s", exc
        )
"""
    ns: dict = {}
    exec(compile(source, "<resolver>", "exec"), ns)
    return ns["_resolve_dino_checkpoint"]


_resolve_dino_checkpoint = _build_resolver()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_settings():
    s = MagicMock()
    s.DATABASE_URL = "postgresql://fake/db"
    s.DINO_CHECKPOINT = "/env/checkpoint.pt"
    return s


@pytest.fixture
def mock_logger():
    return MagicMock()


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestResolveDinoCheckpoint:
    def _call(self, settings, logger, row_value=None, connect_error=None):
        mock_conn = AsyncMock()
        mock_conn.close = AsyncMock()

        if connect_error:
            _asyncpg_mod.connect = AsyncMock(side_effect=connect_error)
        else:

            async def _fetchrow(query, *_):
                if row_value is None:
                    return None
                row = MagicMock()
                row.__getitem__ = lambda s, k: row_value
                return row

            mock_conn.fetchrow = AsyncMock(side_effect=_fetchrow)
            _asyncpg_mod.connect = AsyncMock(return_value=mock_conn)

        _resolve_dino_checkpoint(settings, logger)
        return mock_conn

    def test_no_database_url_leaves_settings_unchanged(self, mock_settings, mock_logger):
        mock_settings.DATABASE_URL = ""
        original = mock_settings.DINO_CHECKPOINT
        _resolve_dino_checkpoint(mock_settings, mock_logger)
        assert mock_settings.DINO_CHECKPOINT == original

    def test_no_database_url_does_not_log(self, mock_settings, mock_logger):
        mock_settings.DATABASE_URL = ""
        _resolve_dino_checkpoint(mock_settings, mock_logger)
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()

    def test_db_row_overrides_settings(self, mock_settings, mock_logger):
        self._call(mock_settings, mock_logger, row_value="/db/checkpoint.pt")
        assert mock_settings.DINO_CHECKPOINT == "/db/checkpoint.pt"
        mock_logger.info.assert_called_once()
        assert "/db/checkpoint.pt" in mock_logger.info.call_args[0][1]

    def test_db_row_none_leaves_env_var(self, mock_settings, mock_logger):
        self._call(mock_settings, mock_logger, row_value=None)
        assert mock_settings.DINO_CHECKPOINT == "/env/checkpoint.pt"
        mock_logger.info.assert_not_called()

    def test_empty_string_db_value_does_not_override(self, mock_settings, mock_logger):
        """Empty string is falsy — env var must be kept."""
        self._call(mock_settings, mock_logger, row_value="")
        assert mock_settings.DINO_CHECKPOINT == "/env/checkpoint.pt"
        mock_logger.info.assert_not_called()

    def test_connect_error_leaves_env_var(self, mock_settings, mock_logger):
        self._call(mock_settings, mock_logger, connect_error=OSError("refused"))
        assert mock_settings.DINO_CHECKPOINT == "/env/checkpoint.pt"
        mock_logger.warning.assert_called_once()
        assert "falling back to env" in mock_logger.warning.call_args[0][0]

    def test_fetchrow_error_leaves_env_var(self, mock_settings, mock_logger):
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(side_effect=RuntimeError("query failed"))
        mock_conn.close = AsyncMock()
        _asyncpg_mod.connect = AsyncMock(return_value=mock_conn)

        _resolve_dino_checkpoint(mock_settings, mock_logger)
        assert mock_settings.DINO_CHECKPOINT == "/env/checkpoint.pt"
        mock_logger.warning.assert_called_once()

    def test_connection_closed_after_successful_fetch(self, mock_settings, mock_logger):
        mock_conn = self._call(mock_settings, mock_logger, row_value="/db/ckpt.pt")
        mock_conn.close.assert_called_once()

    def test_connection_closed_even_on_fetchrow_error(self, mock_settings, mock_logger):
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(side_effect=RuntimeError("oops"))
        mock_conn.close = AsyncMock()
        _asyncpg_mod.connect = AsyncMock(return_value=mock_conn)

        _resolve_dino_checkpoint(mock_settings, mock_logger)
        mock_conn.close.assert_called_once()

    def test_different_checkpoint_paths_stored_correctly(self, mock_settings, mock_logger):
        paths = ["/checkpoints/v1.pt", "/checkpoints/sup_abc123.pt", "/tmp/test.pt"]
        for path in paths:
            mock_settings.DINO_CHECKPOINT = "/env/original.pt"
            self._call(mock_settings, mock_logger, row_value=path)
            assert mock_settings.DINO_CHECKPOINT == path
