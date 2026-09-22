"""Apply small fixes to the pinned ss-fusion wheel used by local runs.

The research pipeline is installed from a published tag. These guarded edits
keep this checkout's local run correct until the fixes are released upstream.
"""

from importlib import reload
from importlib.util import find_spec
from pathlib import Path


def _replace(source: str, old: str, new: str, expected: int = 1) -> str:
    count = source.count(old)
    if count == 0 and new in source:
        return source
    if count != expected:
        raise RuntimeError(f"pinned pipeline changed: expected {expected} copies, found {count}")
    return source.replace(old, new)


def _patch_qwen(source: str) -> str:
    if "def _selected_sidecar(self)" in source:
        return source
    source = _replace(
        source,
        "settings.GEMMA_API_TIMEOUT_SEC if settings.GEMMA_API_URL else settings.QWEN_TIMEOUT_SEC",
        "settings.QWEN_TIMEOUT_SEC",
        2,
    )
    source = _replace(
        source,
        "return bool(settings.GEMMA_API_URL or settings.QWEN_API_URL)",
        "return bool(settings.QWEN_API_URL)",
    )
    source = _replace(
        source,
        "_api_url = settings.GEMMA_API_URL or settings.QWEN_API_URL",
        "_api_url = settings.QWEN_API_URL",
        3,
    )
    source = _replace(
        source,
        "_api_model = settings.GEMMA_API_MODEL if settings.GEMMA_API_URL else settings.QWEN_MODEL",
        "_api_model = settings.QWEN_MODEL",
        2,
    )
    source = _replace(
        source,
        "settings.GEMMA_API_BACKEND if settings.GEMMA_API_URL else settings.QWEN_BACKEND",
        "settings.QWEN_BACKEND",
    )
    source = _replace(
        source,
        'sidecar = "Gemma" if settings.GEMMA_API_URL else "Qwen"',
        'sidecar = "Qwen"',
    )
    source = _replace(
        source,
        "def __init__(self, clip_prescreen_fn: Callable[[Image.Image], bool] | None = None):",
        "def __init__(\n"
        "        self, clip_prescreen_fn: Callable[[Image.Image], bool] | None = None,\n"
        "        *, prefer_qwen: bool = False,\n"
        "    ):",
    )
    source = _replace(
        source,
        "        self._client = None\n\n    # -- Public interface",
        '''        self._client = None
        self._prefer_qwen = prefer_qwen

    def _selected_sidecar(self) -> tuple[str, str, str, int]:
        if self._prefer_qwen or not settings.GEMMA_API_URL:
            return (
                settings.QWEN_API_URL,
                settings.QWEN_MODEL,
                settings.QWEN_BACKEND,
                settings.QWEN_TIMEOUT_SEC,
            )
        return (
            settings.GEMMA_API_URL,
            settings.GEMMA_API_MODEL,
            settings.GEMMA_API_BACKEND,
            settings.GEMMA_API_TIMEOUT_SEC,
        )

    # -- Public interface''',
    )
    source = _replace(
        source,
        "return bool(settings.QWEN_API_URL)",
        "return bool(self._selected_sidecar()[0])",
    )
    source = _replace(
        source,
        "_api_url = settings.QWEN_API_URL\n        _api_model = settings.QWEN_MODEL",
        "_api_url, _api_model, _, _ = self._selected_sidecar()",
        2,
    )
    source = _replace(
        source,
        '''        _api_url = settings.QWEN_API_URL
        backend = (
            settings.QWEN_BACKEND
        ).lower()
        timeout = min(
            settings.QWEN_TIMEOUT_SEC,
            10,
        )''',
        '''        _api_url, _, configured_backend, configured_timeout = self._selected_sidecar()
        backend = configured_backend.lower()
        timeout = min(configured_timeout, 10)''',
    )
    source = _replace(
        source,
        'sidecar = "Qwen"',
        'sidecar = "Qwen" if self._prefer_qwen or not settings.GEMMA_API_URL else "Gemma"',
    )
    return source


def _patch_vram(source: str) -> str:
    if "# Evict sidecars at every CUDA handoff" in source:
        return source
    start = source.index("    if baseline_free_gb >= required:\n")
    end = source.index("    unload_count = _unload_known_sidecars(\n", start)
    source = source[:start] + "    # Evict sidecars at every CUDA handoff, even if free VRAM looks sufficient.\n" + source[end:]
    return source


def _patch_qwen_step(source: str) -> str:
    if "    _unload_ollama_model(settings.QWEN_API_URL, settings.QWEN_MODEL)" in source:
        return _replace(
            source,
            "qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn)",
            "qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn, prefer_qwen=True)",
        )
    source = _replace(
        source,
        "from ..caption_helpers.vram import _log_vram_snapshot",
        "from ..caption_helpers.ollama import _unload_ollama_model\n"
        "from ..caption_helpers.vram import _log_vram_snapshot",
    )
    source = _replace(
        source,
        "qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn)",
        "qwen = QwenModel(clip_prescreen_fn=clip_prescreen_fn, prefer_qwen=True)",
    )
    source = _replace(
        source,
        "    _log_vram_snapshot(\"after Qwen sidecar use\")",
        "    _unload_ollama_model(settings.QWEN_API_URL, settings.QWEN_MODEL)\n"
        "    _log_vram_snapshot(\"after Qwen sidecar use\")",
    )
    return source


def _patch_ollama(source: str) -> str:
    if "Ollama unload verified" in source:
        return source
    start = source.index("def _unload_ollama_model(api_url: str, model: str) -> bool:\n")
    end = source.index("def _unload_known_sidecars(", start)
    replacement = '''def _unload_ollama_model(api_url: str, model: str) -> bool:
    """Unload an Ollama model and verify it is absent from /api/ps."""
    import time

    import httpx

    if not api_url or not model:
        return False
    base = api_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        response = httpx.post(
            f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=15.0
        )
        if response.status_code != 200:
            _log.debug("Ollama unload returned HTTP %d", response.status_code)
            return False
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            resident = httpx.get(f"{base}/api/ps", timeout=5.0)
            resident.raise_for_status()
            names = {item.get("name") for item in resident.json().get("models", [])}
            if model not in names:
                _log.info("  Ollama unload verified: '%s'", model)
                return True
            time.sleep(0.5)
        _log.warning("  Ollama model remained resident after unload: '%s'", model)
    except Exception as exc:
        _log.debug("Could not verify Ollama unload for '%s': %s", model, exc)
    return False


'''
    return source[:start] + replacement + source[end:]


def _patch_caption_report(source: str) -> str:
    return _replace(
        source,
        'f"Frames captioned: {len(caption_results)}  |  Unique scenes: {n_segments}"',
        'f"Frames captioned: {sum(bool(r.get(\'caption\')) for r in caption_results)}"'
        'f"/{len(caption_results)}  |  Unique scenes: {n_segments}"',
    )


def _patch_distill_inputs(source: str) -> str:
    if "_valid_caption_indices" in source:
        return source
    old = '''                _cap_texts = [r.get("caption") or "" for r in caption_results]
                _cap_texts = [t for t in _cap_texts if t.strip()]
                if _cap_texts:
                    _clip_model = models["clip"]
                    if hasattr(_clip_model, "encode_texts") and not (
                        _HAS_GEMMA and isinstance(_clip_model, GemmaEmbedder)
                    ):
                        _cap_anchor_embs = _clip_model.encode_texts(_cap_texts)
'''
    new = '''                _cap_texts = [r.get("caption") or "" for r in caption_results]
                _valid_caption_indices = [i for i, text in enumerate(_cap_texts) if text.strip()]
                if _valid_caption_indices:
                    _clip_model = models["clip"]
                    if hasattr(_clip_model, "encode_texts") and not (
                        _HAS_GEMMA and isinstance(_clip_model, GemmaEmbedder)
                    ):
                        _valid_embs = _clip_model.encode_texts(
                            [_cap_texts[i] for i in _valid_caption_indices]
                        )
                        _cap_anchor_embs = np.zeros(
                            (len(_cap_texts), _valid_embs.shape[1]), dtype=_valid_embs.dtype
                        )
                        _cap_anchor_embs[_valid_caption_indices] = _valid_embs
'''
    return _replace(source, old, new)


def _patch_graph_distill_inputs(source: str) -> str:
    if "_valid_caption_indices" in source:
        return source
    old = '''            _cap_texts = [
                r.get("caption", "") for r in caption_results if r.get("caption", "").strip()
            ]
            clip_model = models["clip"]
            if _cap_texts and hasattr(clip_model, "encode_texts"):
                _cap_anchor_embs = clip_model.encode_texts(_cap_texts)
'''
    new = '''            import numpy as np

            _cap_texts = [r.get("caption") or "" for r in caption_results]
            _valid_caption_indices = [i for i, text in enumerate(_cap_texts) if text.strip()]
            clip_model = models["clip"]
            if _valid_caption_indices and hasattr(clip_model, "encode_texts"):
                _valid_embs = clip_model.encode_texts(
                    [_cap_texts[i] for i in _valid_caption_indices]
                )
                _cap_anchor_embs = np.zeros(
                    (len(_cap_texts), _valid_embs.shape[1]), dtype=_valid_embs.dtype
                )
                _cap_anchor_embs[_valid_caption_indices] = _valid_embs
'''
    return _replace(source, old, new)


def _patch_distill_training(source: str) -> str:
    if "class _IndexedFrameDataset(Dataset):" in source:
        return source
    source = _replace(
        source,
        "# -- RKD loss functions --------------------------------------------------------",
        '''class _IndexedFrameDataset(Dataset):
    def __init__(self, frames: _FrameDataset) -> None:
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.frames[idx], idx


# -- RKD loss functions --------------------------------------------------------''',
    )
    source = _replace(
        source,
        "        loader = DataLoader(\n            dataset,",
        "        loader = DataLoader(\n            _IndexedFrameDataset(dataset),",
    )
    source = _replace(
        source,
        "            for batch_idx, batch in enumerate(loader):\n                batch = batch.to(self.device)",
        "            for batch, sample_indices in loader:\n                batch = batch.to(self.device)",
    )
    source = _replace(
        source,
        "        trainable = list(self.student.parameters()) + list(self._proj.parameters())",
        '''        trainable = list(self.student.parameters()) + list(self._proj.parameters())
        if cfg.lambda_caption_anchor > 0 and cfg.caption_embeddings is not None:
            self._cap_proj = nn.Linear(
                self._s_dim, cfg.caption_embeddings.shape[1], bias=False
            ).to(self.device)
            nn.init.orthogonal_(self._cap_proj.weight)
            trainable += list(self._cap_proj.parameters())''',
    )
    old = '''                    # Align caption anchors to this batch by dataset index
                    start = batch_idx * cfg.batch_size
                    end = min(start + batch.shape[0], len(_cap_anchors))
                    if end > start:
                        cap_batch = _cap_anchors[start:end]  # (B', clip_dim)
                        s_for_cap = s_raw[: end - start]  # (B', s_dim)
                        # Project student to caption dim if dims differ
                        if not hasattr(self, "_cap_proj"):
                            self._cap_proj = nn.Linear(
                                s_raw.shape[1], cap_batch.shape[1], bias=False
                            ).to(self.device)
                            nn.init.orthogonal_(self._cap_proj.weight)
                        s_cap_proj = F.normalize(self._cap_proj(s_for_cap.float()), dim=-1)
                        cap_norm = F.normalize(cap_batch, dim=-1)
                        l_caption = (1.0 - (s_cap_proj * cap_norm).sum(dim=-1)).mean()
                        loss = loss + cfg.lambda_caption_anchor * l_caption
'''
    new = '''                    # Match each shuffled frame to its own non-empty caption anchor.
                    cap_batch = _cap_anchors[sample_indices.to(self.device)]
                    valid = cap_batch.abs().sum(dim=-1) > 0
                    if valid.any():
                        s_cap_proj = F.normalize(
                            self._cap_proj(s_raw[valid].float()), dim=-1
                        )
                        cap_norm = F.normalize(cap_batch[valid], dim=-1)
                        l_caption = (1.0 - (s_cap_proj * cap_norm).sum(dim=-1)).mean()
                        loss = loss + cfg.lambda_caption_anchor * l_caption
'''
    return _replace(source, old, new)


def _patch_scene_probes(source: str) -> str:
    if '"computer screen showing a map or mission dashboard"' not in source:
        source = _replace(
            source,
            '''    "coastal or water feature",
]''',
            '''    "coastal or water feature",
    "aircraft parked or flying at an airport",
    "computer screen showing a map or mission dashboard",
    "promotional presentation with text and graphics",
    "indoor meeting or military briefing",
]''',
        )
    if "Scene mix:" in source:
        return source
    source = _replace(
        source,
        '''        if clf.get("category_distribution"):
            self.scene_type = next(iter(clf["category_distribution"]), "")''',
        '''        distribution = clf.get("category_distribution") or {}
        if distribution:
            top_category, top_count = next(iter(distribution.items()))
            total = max(1, int(clf.get("n_frames", sum(distribution.values()))))
            if top_count / total < 0.6:
                self.scene_type = "Scene mix: " + ", ".join(list(distribution)[:3])
            else:
                self.scene_type = top_category''',
    )
    return _replace(
        source,
        'parts.append(f"Dominant scene: {self.scene_type}")',
        '''parts.append(
                self.scene_type if self.scene_type.startswith("Scene mix:")
                else f"Dominant scene: {self.scene_type}"
            )''',
    )


def _patch_phase4_handoff(source: str) -> str:
    if "# Release synthesis VLM before reasoning audit." in source:
        return source
    return _replace(
        source,
        '''        step_video_synthesis(
            video_name, video_dir, video_context, api_url=_qwen_url, model=_qwen_model
        )
    _append_agentic_step(''',
        '''        step_video_synthesis(
            video_name, video_dir, video_context, api_url=_qwen_url, model=_qwen_model
        )
    # Release synthesis VLM before reasoning audit.
    if device == "cuda":
        _unload_known_sidecars([(_qwen_url, _qwen_model)])
    _append_agentic_step(''',
    )


def _patch_graph_phase4_handoff(source: str) -> str:
    if "# Release synthesis VLM before reasoning audit." in source:
        return source
    return _replace(
        source,
        '''    stats.setdefault("timings", {})["Z_synthesis"] = time.monotonic() - t0

    _append_agentic_step(''',
        '''    stats.setdefault("timings", {})["Z_synthesis"] = time.monotonic() - t0
    # Release synthesis VLM before reasoning audit.
    if device == "cuda":
        from ...steps.caption import _unload_known_sidecars

        _unload_known_sidecars([(_qwen_url, _qwen_model)])

    _append_agentic_step(''',
    )


def _patch_finetuned_search(source: str) -> str:
    if "Re-embed the gallery with the adapted backbone" in source:
        return source
    start = source.index("def step_finetuned_model_search_test(\n")
    replacement = '''def step_finetuned_model_search_test(
    frame_list: list[tuple[str, float]],
    store: Any,
    is_qdrant: bool,
    models: dict[str, Any],
    query_frame: str,
    query_t_sec: float,
    video_id: str,
    video_name: str,
    video_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    """Evaluate the SSL checkpoint against a gallery in the same embedding space."""
    import torch

    from ..report import write_search_md

    checkpoint = video_dir / "checkpoints" / "dino_ssl_best.pt"
    embedder = models.get("dino")
    if embedder is None or not checkpoint.is_file():
        _log.warning("Fine-tuned search unavailable: DINO embedder or checkpoint missing")
        return {"results": [], "infer_ms": 0.0}

    # Re-embed the gallery with the adapted backbone; the base store cannot be
    # searched with an adapted query because its vectors use the base weights.
    backbone = embedder.model
    base_weights = {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()}
    t0 = time.time()
    try:
        adapted = torch.load(checkpoint, map_location="cpu", weights_only=True)
        backbone.load_state_dict(adapted)
        backbone.eval()
        chunks = []
        for offset in range(0, len(frame_list), 16):
            images = []
            for frame_path, _ in frame_list[offset : offset + 16]:
                with Image.open(frame_path) as image:
                    images.append(image.convert("RGB"))
            chunks.append(embedder.encode_images(images))
        vectors = np.vstack(chunks)
    finally:
        backbone.load_state_dict(base_weights)
        backbone.eval()

    query_index = next(
        (index for index, (path, _) in enumerate(frame_list) if path == query_frame), None
    )
    if query_index is None:
        raise ValueError("query frame is absent from the fine-tuned gallery")
    scores = vectors @ vectors[query_index]
    eligible = [
        index for index, (path, timestamp) in enumerate(frame_list)
        if path != query_frame and abs(timestamp - query_t_sec) >= 1.0
    ]
    ranked = sorted(eligible, key=lambda index: float(scores[index]), reverse=True)[:top_k]
    results = [
        {"score": float(scores[index]),
         "payload": {"frame_path": frame_list[index][0], "t_sec": frame_list[index][1]}}
        for index in ranked
    ]
    write_search_md(
        video_dir / "finetuned_search.md", video_name, "Fine-tuned DINOv3 (SSL adapted)",
        query_frame, results, query_t_sec,
    )
    elapsed = time.time() - t0
    return {"results": results, "infer_ms": elapsed * 1000 / max(len(frame_list), 1)}
'''
    return source[:start] + replacement


def _patch_comparison_report(source: str) -> str:
    if "Cosine values from different embedding spaces" in source:
        return source
    source = _replace(source, '"## Search Quality Comparison",', '"## Retrieval Diagnostics",')
    return _replace(
        source,
        '''        f"| Δ score | — | {avg_ft - avg_base:+.4f} |",
        f"| Result overlap | {overlap}/{len(base_results)} frames in common | |",
        "",
        "## Model Statistics",''',
        '''        f"| Result overlap | {overlap}/{len(base_results)} frames in common | |",
        "",
        "Cosine values from different embedding spaces are not directly comparable. "
        "Use ranked neighbors or held-out labels to judge retrieval quality.",
        "",
        "## Model Statistics",''',
    )


_PATCHES = (
    ("selfsuvis.pipeline.vision.qwen", _patch_qwen),
    ("ssv_vdp.steps.caption_helpers.ollama", _patch_ollama),
    ("ssv_vdp.steps.caption_helpers.vram", _patch_vram),
    ("ssv_vdp.steps.caption._vlm", _patch_qwen_step),
    ("ssv_vdp.steps.report_helpers._captions", _patch_caption_report),
    ("ssv_vdp.pipeline.runner_helpers._pipeline_phase3", _patch_distill_inputs),
    ("ssv_vdp.pipeline.nodes.phase3_ssl", _patch_graph_distill_inputs),
    ("selfsuvis.pipeline.training.distill", _patch_distill_training),
    ("ssv_vdp.steps.common", _patch_scene_probes),
    ("ssv_vdp.pipeline.runner_helpers._pipeline_phase4", _patch_phase4_handoff),
    ("ssv_vdp.pipeline.nodes.phase4", _patch_graph_phase4_handoff),
    ("ssv_vdp.steps.perception.embed", _patch_finetuned_search),
    ("ssv_vdp.steps.report_helpers._search", _patch_comparison_report),
)


def main() -> None:
    for module_name, patch in _PATCHES:
        spec = find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"required local pipeline module is missing: {module_name}")
        path = Path(spec.origin)
        before = path.read_text()
        after = patch(before)
        if before != after:
            path.write_text(after)
            print(f"patched {module_name}")

    # Preflight chooses models from free VRAM, so release last run's sidecars
    # before it samples the GPU. The pipeline repeats this check at startup.
    from selfsuvis.pipeline.core.config import settings
    from ssv_vdp.steps.caption_helpers import ollama

    reload(ollama)
    ollama._unload_known_sidecars(
        [
            (settings.GEMMA_API_URL, settings.GEMMA_API_MODEL),
            (settings.QWEN_API_URL, settings.QWEN_MODEL),
            (settings.REASONING_API_URL, settings.REASONING_MODEL),
        ]
    )


if __name__ == "__main__":
    main()
