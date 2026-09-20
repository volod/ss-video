"""Schema bootstrap helpers stay importable without loading layered env."""

from selfsuvis.fusion_rt.migrate import FUSION_SCHEMA
from selfsuvis.fusion_rt.migrate import apply_schema as apply_fusion_schema
from selfsuvis.pipeline.storage.migrate_video import VIDEO_SCHEMA
from selfsuvis.pipeline.storage.migrate_video import apply_schema as apply_video_schema
from selfsuvis.scripts.migrate_postgres import main, migrate


class _FakeConn:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, stmt: str, *_args) -> None:
        self.statements.append(stmt)

    async def close(self) -> None:
        return None


class UniqueViolationError(Exception):
    """Name-matched stand-in for asyncpg.exceptions.UniqueViolationError."""


async def test_apply_video_schema_runs_every_statement() -> None:
    conn = _FakeConn()
    await apply_video_schema(conn, verbose=False)
    assert len(conn.statements) == len(VIDEO_SCHEMA)


async def test_apply_fusion_schema_runs_every_statement() -> None:
    conn = _FakeConn()
    await apply_fusion_schema(conn, verbose=False)
    assert len(conn.statements) == len(FUSION_SCHEMA)


async def test_apply_schema_ignores_create_unique_violation() -> None:
    class _Conn:
        async def execute(self, stmt: str, *_args) -> None:
            if stmt.upper().lstrip().startswith("CREATE"):
                raise UniqueViolationError("duplicate key")

    await apply_video_schema(_Conn(), verbose=False)
    await apply_fusion_schema(_Conn(), verbose=False)


async def test_migrate_all_uses_video_and_fusion_urls(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _video(url: str, *, verbose: bool = True) -> None:
        captured["video"] = url

    async def _fusion(url: str, *, verbose: bool = True) -> None:
        captured["fusion"] = url

    monkeypatch.setattr("selfsuvis.pipeline.storage.migrate_video.migrate", _video)
    monkeypatch.setattr("selfsuvis.fusion_rt.migrate.migrate", _fusion)
    video_url = "postgresql://selfsuvis:selfsuvis@postgres:5432/selfsuvis"
    await migrate(video_url, verbose=False)
    assert captured["video"] == video_url
    assert captured["fusion"] == "postgresql://selfsuvis:selfsuvis@postgres:5432/selfsuvis_fusion"


async def test_migrate_owner_video_skips_fusion(monkeypatch) -> None:
    captured: list[str] = []

    async def _video_connect(url: str):
        captured.append(url)
        return _FakeConn()

    async def _boom(url: str, *, verbose: bool = True) -> None:
        raise AssertionError("fusion migrate must not run")

    monkeypatch.setattr("selfsuvis.pipeline.storage.migrate_video.asyncpg.connect", _video_connect)
    monkeypatch.setattr("selfsuvis.fusion_rt.migrate.migrate", _boom)
    await migrate(
        "postgresql://selfsuvis:selfsuvis@postgres:5432/selfsuvis", verbose=False, owner="video"
    )
    assert captured == ["postgresql://selfsuvis:selfsuvis@postgres:5432/selfsuvis"]


def test_ssv_migrate_entry_point_exists() -> None:
    assert callable(main)
