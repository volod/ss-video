"""Unit tests for GET /admin/automation-roi.

All DB calls are mocked — no live PostgreSQL required.
asyncpg is not installed in the test environment; it is stubbed in sys.modules.
"""

import asyncio
import json
import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

# ── Stub asyncpg before any imports that use it ───────────────────────────────
if "asyncpg" not in sys.modules:
    _asyncpg = types.ModuleType("asyncpg")
    _asyncpg.connect = MagicMock()  # patch.object requires attribute to pre-exist
    _asyncpg.Pool = MagicMock()  # type: ignore[attr-defined]
    sys.modules["asyncpg"] = _asyncpg

# ── Stub app.state (heavy deps) ───────────────────────────────────────────────
if "selfsuvis.app.state" not in sys.modules:
    _state_stub = MagicMock()
    sys.modules["selfsuvis.app.state"] = _state_stub


class _Resp:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


def _mock_conn(
    total_annotated=0,
    first_at=None,
    last_at=None,
    campaigns=0,
    ft_rows=None,
    reembed_done=0,
):
    conn = AsyncMock()
    conn.close = AsyncMock()

    if ft_rows is None:
        ft_rows = []

    async def _fetchval(query, *args):
        q = query.strip().lower()
        if "count(*) from frames" in q:
            return total_annotated
        if "count(distinct" in q:
            return campaigns
        if "count(*) from jobs" in q:
            return reembed_done
        return 0

    async def _fetchrow(query, *args):
        if "min(" in query.lower():
            if first_at is None:
                return None
            row = MagicMock()
            row.__getitem__ = lambda s, k: first_at if k == "first_at" else last_at
            return row
        return None

    async def _fetch(query, *args):
        if "supervised_finetune" in query:
            rows = []
            for r in ft_rows:
                m = MagicMock()
                m.__getitem__ = lambda s, k, _r=r: "finished" if k == "status" else json.dumps(_r)
                rows.append(m)
            return rows
        return []

    conn.fetchval = AsyncMock(side_effect=_fetchval)
    conn.fetchrow = AsyncMock(side_effect=_fetchrow)
    conn.fetch = AsyncMock(side_effect=_fetch)
    return conn


class TestAutomationROI:
    def _get(self, conn, db_url="postgresql://fake/db"):
        from selfsuvis.app.routers import admin as admin_mod

        with (
            patch.object(sys.modules["asyncpg"], "connect", AsyncMock(return_value=conn)),
            patch.object(admin_mod.settings, "DATABASE_URL", db_url),
        ):
            return _Resp(asyncio.run(admin_mod.automation_roi()))

    # ── no-data cases ─────────────────────────────────────────────────────────

    def test_insufficient_data_when_no_annotations(self):
        conn = _mock_conn(total_annotated=0, first_at=None)
        resp = self._get(conn)
        assert resp.status_code == 200
        body = resp.json()
        assert body["verdict"] == "INSUFFICIENT_DATA"
        assert body["total_annotated_frames"] == 0
        assert body["days_observed"] is None
        assert body["annotation_frequency_per_week"] is None

    def test_insufficient_data_when_less_than_7_days(self):
        now = time.time()
        conn = _mock_conn(
            total_annotated=5,
            first_at=now - 3 * 86400,
            last_at=now,
            campaigns=1,
        )
        resp = self._get(conn)
        body = resp.json()
        assert body["verdict"] == "INSUFFICIENT_DATA"
        assert body["days_observed"] is not None
        assert body["annotation_frequency_per_week"] is None

    # ── verdict bands ─────────────────────────────────────────────────────────

    def test_low_frequency_verdict(self):
        now = time.time()
        # 2 annotations over 30 days → ~0.47/week < 0.5
        conn = _mock_conn(
            total_annotated=2,
            first_at=now - 30 * 86400,
            last_at=now,
            campaigns=1,
        )
        resp = self._get(conn)
        body = resp.json()
        assert body["verdict"] == "LOW_FREQUENCY"
        assert body["annotation_frequency_per_week"] < 0.5

    def test_moderate_frequency_verdict(self):
        now = time.time()
        # 4 annotations over 28 days → 1.0/week → MODERATE
        conn = _mock_conn(
            total_annotated=4,
            first_at=now - 28 * 86400,
            last_at=now,
            campaigns=2,
        )
        resp = self._get(conn)
        body = resp.json()
        assert body["verdict"] == "MODERATE_FREQUENCY"

    def test_high_frequency_verdict(self):
        now = time.time()
        # 20 annotations over 7 days → 20/week > 2.0
        conn = _mock_conn(
            total_annotated=20,
            first_at=now - 7 * 86400,
            last_at=now,
            campaigns=3,
        )
        resp = self._get(conn)
        body = resp.json()
        assert body["verdict"] == "HIGH_FREQUENCY"
        assert body["annotation_frequency_per_week"] > 2.0

    # ── finetune metrics ──────────────────────────────────────────────────────

    def test_finetune_counts_accepted(self):
        now = time.time()
        ft_rows = [
            {"accepted": True},
            {"accepted": True},
            {"accepted": False},
        ]
        conn = _mock_conn(
            total_annotated=50,
            first_at=now - 30 * 86400,
            last_at=now,
            campaigns=2,
            ft_rows=ft_rows,
        )
        resp = self._get(conn)
        body = resp.json()
        assert body["finetune_jobs_triggered"] == 3
        assert body["finetune_jobs_accepted"] == 2
        assert abs(body["finetune_acceptance_rate"] - 2 / 3) < 0.01
        assert body["model_reloads"] == 2

    def test_ops_minutes_saved_calculation(self):
        now = time.time()
        # 3 accepted × 3 min = 9 min
        ft_rows = [{"accepted": True}] * 3
        conn = _mock_conn(
            total_annotated=30,
            first_at=now - 21 * 86400,
            last_at=now,
            ft_rows=ft_rows,
        )
        resp = self._get(conn)
        assert resp.json()["estimated_ops_minutes_saved"] == 9

    def test_acceptance_rate_none_when_no_jobs(self):
        now = time.time()
        conn = _mock_conn(
            total_annotated=10,
            first_at=now - 14 * 86400,
            last_at=now,
            ft_rows=[],
        )
        resp = self._get(conn)
        body = resp.json()
        assert body["finetune_acceptance_rate"] is None
        assert body["estimated_ops_minutes_saved"] == 0

    def test_reembed_sweeps_counted(self):
        now = time.time()
        conn = _mock_conn(
            total_annotated=20,
            first_at=now - 14 * 86400,
            last_at=now,
            reembed_done=4,
        )
        resp = self._get(conn)
        assert resp.json()["reembed_sweeps_completed"] == 4

    # ── error handling ────────────────────────────────────────────────────────

    def test_no_database_url_returns_error(self):
        conn = _mock_conn()
        resp = self._get(conn, db_url="")
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_db_connect_error_returns_error(self):
        from selfsuvis.app.routers import admin as admin_mod

        with (
            patch.object(
                sys.modules["asyncpg"], "connect", AsyncMock(side_effect=OSError("refused"))
            ),
            patch.object(admin_mod.settings, "DATABASE_URL", "postgresql://fake/db"),
        ):
            resp = _Resp(asyncio.run(admin_mod.automation_roi()))
        assert resp.status_code == 200
        assert "error" in resp.json()

    # ── response shape ────────────────────────────────────────────────────────

    def test_response_contains_all_expected_keys(self):
        now = time.time()
        conn = _mock_conn(
            total_annotated=10,
            first_at=now - 14 * 86400,
            last_at=now,
            campaigns=1,
            ft_rows=[{"accepted": True}],
            reembed_done=1,
        )
        body = self._get(conn).json()
        expected = {
            "total_annotated_frames",
            "annotation_campaigns",
            "finetune_jobs_triggered",
            "finetune_jobs_accepted",
            "finetune_acceptance_rate",
            "model_reloads",
            "reembed_sweeps_completed",
            "estimated_ops_minutes_saved",
            "first_annotation_at",
            "last_annotation_at",
            "days_observed",
            "annotation_frequency_per_week",
            "verdict",
            "verdict_detail",
        }
        assert expected.issubset(body.keys())

    def test_verdict_detail_is_non_empty_string(self):
        now = time.time()
        conn = _mock_conn(
            total_annotated=5,
            first_at=now - 30 * 86400,
            last_at=now,
        )
        body = self._get(conn).json()
        assert isinstance(body["verdict_detail"], str)
        assert len(body["verdict_detail"]) > 10
