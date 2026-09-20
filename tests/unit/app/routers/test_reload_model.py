"""Unit tests for POST /admin/reload-model.

All filesystem I/O and app.state references are mocked — no GPU or real
checkpoint files required.
"""

import sys
import types
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

# ── Stub asyncpg (not installed in test venv) ─────────────────────────────────
if "asyncpg" not in sys.modules:
    _asyncpg = types.ModuleType("asyncpg")
    _asyncpg.connect = MagicMock()
    _asyncpg.Pool = MagicMock()  # type: ignore[attr-defined]
    sys.modules["asyncpg"] = _asyncpg

# ── Stub app.state (heavy model loading side-effects) ────────────────────────
# Must be in sys.modules before admin router is imported.
if "selfsuvis.app.state" not in sys.modules:
    sys.modules["selfsuvis.app.state"] = MagicMock()


# ── Test client factory ───────────────────────────────────────────────────────


def _make_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from selfsuvis.app.routers.admin import router as admin_router

    app = FastAPI()
    app.include_router(admin_router, dependencies=[])
    return TestClient(app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _state_with_dino(dino_model=None, lock_locked=False):
    """Return a mock app.state with configurable dino_model and lock."""
    state = MagicMock()
    state.dino_model = dino_model
    lock = MagicMock()
    lock.locked = MagicMock(return_value=lock_locked)
    lock.__aenter__ = AsyncMock(return_value=None)
    lock.__aexit__ = AsyncMock(return_value=False)
    state.dino_model_lock = lock
    return state


@contextmanager
def _patch_state_and_settings(state, tmp_path):
    """Patch sys.modules['selfsuvis.app.state'] and settings.SUP_CHECKPOINT_DIR together."""
    with (
        patch.dict(sys.modules, {"selfsuvis.app.state": state}),
        patch("selfsuvis.app.routers.admin.settings") as mock_settings,
    ):
        mock_settings.SUP_CHECKPOINT_DIR = str(tmp_path)
        yield mock_settings


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestReloadModel:
    def test_dino_not_loaded_returns_400(self, tmp_path):
        """400 when MODEL_NAME is not dinov2/dinov3 (dino_model is None)."""
        state = _state_with_dino(dino_model=None)
        client = _make_client()
        with _patch_state_and_settings(state, tmp_path):
            resp = client.post("/admin/reload-model", json={})
        assert resp.status_code == 400
        assert "not loaded" in resp.json()["detail"].lower()

    def test_no_checkpoint_and_no_active_txt_returns_400(self, tmp_path):
        """400 when no checkpoint given and active_checkpoint.txt is absent."""
        state = _state_with_dino(dino_model=MagicMock())
        client = _make_client()
        with _patch_state_and_settings(state, tmp_path / "ckpts"):
            resp = client.post("/admin/reload-model", json={})
        assert resp.status_code == 400
        assert "active_checkpoint.txt" in resp.json()["detail"]

    def test_checkpoint_file_not_found_returns_400(self, tmp_path):
        """400 when the specified checkpoint path does not exist on disk."""
        state = _state_with_dino(dino_model=MagicMock())
        client = _make_client()
        with _patch_state_and_settings(state, tmp_path):
            resp = client.post(
                "/admin/reload-model",
                json={"checkpoint": str(tmp_path / "missing.pt")},
            )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    def test_lock_busy_returns_409(self, tmp_path):
        """409 when another reload is already in progress."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")
        state = _state_with_dino(dino_model=MagicMock(), lock_locked=True)
        client = _make_client()
        with _patch_state_and_settings(state, tmp_path):
            resp = client.post("/admin/reload-model", json={"checkpoint": str(ckpt)})
        assert resp.status_code == 409
        assert "in progress" in resp.json()["detail"].lower()

    def test_successful_reload_returns_ok(self, tmp_path):
        """200 with status=ok and the checkpoint path on success."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")
        dino = MagicMock()
        dino.load_backbone_checkpoint = MagicMock()
        state = _state_with_dino(dino_model=dino)
        client = _make_client()
        with _patch_state_and_settings(state, tmp_path):
            resp = client.post("/admin/reload-model", json={"checkpoint": str(ckpt)})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checkpoint"] == str(ckpt)
        dino.load_backbone_checkpoint.assert_called_once_with(str(ckpt))

    def test_successful_reload_writes_active_checkpoint_txt(self, tmp_path):
        """active_checkpoint.txt is updated on successful reload."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")
        dino = MagicMock()
        dino.load_backbone_checkpoint = MagicMock()
        state = _state_with_dino(dino_model=dino)
        client = _make_client()
        with _patch_state_and_settings(state, tmp_path):
            client.post("/admin/reload-model", json={"checkpoint": str(ckpt)})
        active_txt = tmp_path / "active_checkpoint.txt"
        assert active_txt.exists()
        assert active_txt.read_text().strip() == str(ckpt)

    def test_load_exception_returns_500(self, tmp_path):
        """500 when load_backbone_checkpoint raises (model unchanged)."""
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"fake")
        dino = MagicMock()
        dino.load_backbone_checkpoint = MagicMock(side_effect=RuntimeError("corrupt weights"))
        state = _state_with_dino(dino_model=dino)
        client = _make_client()
        with _patch_state_and_settings(state, tmp_path):
            resp = client.post("/admin/reload-model", json={"checkpoint": str(ckpt)})
        assert resp.status_code == 500
        assert "corrupt weights" in resp.json()["detail"]

    def test_active_checkpoint_txt_used_when_no_body_checkpoint(self, tmp_path):
        """When no checkpoint in body, active_checkpoint.txt is read and used."""
        ckpt = tmp_path / "from_txt.pt"
        ckpt.write_bytes(b"fake")
        active_txt = tmp_path / "active_checkpoint.txt"
        active_txt.write_text(str(ckpt))
        dino = MagicMock()
        dino.load_backbone_checkpoint = MagicMock()
        state = _state_with_dino(dino_model=dino)
        client = _make_client()
        with _patch_state_and_settings(state, tmp_path):
            resp = client.post("/admin/reload-model", json={})
        assert resp.status_code == 200
        dino.load_backbone_checkpoint.assert_called_once_with(str(ckpt))
