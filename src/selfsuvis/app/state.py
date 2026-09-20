"""Application-level singletons loaded lazily so that importing app.main
in CI (e.g. for OpenAPI export) never triggers model downloads or Qdrant
connections.

Usage::

    from selfsuvis.app.state import app_state
    model = app_state.clip_model   # loads on first access
    store = app_state.store        # connects to Qdrant on first access
"""

import asyncio

from selfsuvis.pipeline.core import get_logger, settings, validate_settings

logger = get_logger(__name__)

validate_settings()


class _AppState:
    def __init__(self) -> None:
        self._clip_model = None
        self._dino_model = None
        self._store = None

    def _resolve_dino_checkpoint(self) -> None:
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

    @property
    def clip_model(self):
        if self._clip_model is None:
            from PIL import Image

            if settings.MAX_IMAGE_PIXELS > 0:
                Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

            if settings.MODEL_NAME == "gemma":
                from selfsuvis.models.gemma_model import GemmaEmbedder

                self._clip_model = GemmaEmbedder(
                    model_id=settings.GEMMA_MODEL_ID,
                    device=settings.DEVICE,
                    use_bf16=settings.GEMMA_USE_BF16,
                )
            else:
                from selfsuvis.models.openclip_model import OpenCLIPEmbedder

                self._clip_model = OpenCLIPEmbedder()
        return self._clip_model

    @property
    def dino_model(self):
        if self._dino_model is None and settings.MODEL_NAME in {"dinov2", "dinov3"}:
            self._resolve_dino_checkpoint()
            try:
                from selfsuvis.models.dino_model import DINOEmbedder
                from selfsuvis.pipeline.core import get_dino_model_name

                dino_name = get_dino_model_name(settings.MODEL_NAME)
                if dino_name is None:
                    raise ValueError(f"Unsupported DINO model family: {settings.MODEL_NAME}")
                self._dino_model = DINOEmbedder(dino_name)
            except (RuntimeError, OSError, ValueError) as exc:
                logger.exception("DINO model failed to load, disabling: %s", exc)
                self._dino_model = None
        return self._dino_model

    @property
    def store(self):
        if self._store is None:
            from selfsuvis.pipeline.storage import QdrantStore

            self._store = QdrantStore(
                clip_dim=self.clip_model.image_dim(),
                dino_dim=self.dino_model.image_dim() if self.dino_model else None,
            )
        return self._store


app_state = _AppState()

# Module-level aliases used by existing routers and tests.
# These remain lazy — they resolve on first attribute access.


def _get_clip_model():
    return app_state.clip_model


def _get_dino_model():
    return app_state.dino_model


def _get_store():
    return app_state.store


# Thin proxies so that `from selfsuvis.app.state import clip_model, store` still
# triggers a load (existing callers assign these names to local vars and call
# methods; returning the object here is equivalent).
class _LazyAttr:
    def __init__(self, loader):
        self._loader = loader

    def __getattr__(self, name):
        return getattr(self._loader(), name)

    def __repr__(self):
        return repr(self._loader())


clip_model = _LazyAttr(_get_clip_model)
dino_model = _LazyAttr(_get_dino_model)
store = _LazyAttr(_get_store)
qdrant_store = store  # alias

# Locks used by admin reload and fine-tune (GIL-atomic assignment is safe).
dino_model_lock: asyncio.Lock = asyncio.Lock()
_finetune_lock: asyncio.Lock = asyncio.Lock()
