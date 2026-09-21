"""Grounding, counting, and mask providers.

Neural adapters call the pinned ss-perception wrappers or Hugging Face
pipelines already used by this repository. They load lazily. CountGD is a
count expert: its hits are marked count-only and never become track ids.
Production instantiates the pinned pair, not every candidate.
"""

import hashlib
import importlib
from dataclasses import dataclass
from typing import Protocol

from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal
from selfsuvis.pipeline.analysis4d.pin import ProviderProbe
from selfsuvis.pipeline.analysis4d.tracks import Detection

_PREPROCESS = "analysis4d-providers-v1"
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-tiny"


@dataclass(frozen=True)
class Prompt:
    """Operator text or exemplar prompt. The raw phrase is preserved on each hit."""

    prompt_id: str
    text: str
    normalized: str
    negative: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = ()


@dataclass(frozen=True)
class CountHit:
    """A count for one label. This is not a track."""

    prompt_id: str
    label_normalized: str
    count: int


class GroundingProvider(Protocol):
    """Open-vocabulary or closed-set boxes at a keyframe."""

    provider_id: str

    def ground(self, frame: FrameSignal, prompts: list[Prompt]) -> list[Detection]:
        """Return detections for ``frame``. Count-only providers return none."""


class CountProvider(Protocol):
    """Text or exemplar counts. Disagreement is a signal, not a new identity."""

    provider_id: str

    def count(self, frame: FrameSignal, prompts: list[Prompt]) -> list[CountHit]:
        """Return label counts for ``frame``."""


class MaskProvider(Protocol):
    """Mask refinement and a reset hook for hard cuts."""

    provider_id: str

    def reset(self, reason: str) -> None:
        """Drop memory for the current epoch."""

    def refine(self, frame: FrameSignal, detections: list[Detection]) -> list[Detection]:
        """Return detections with mask bits filled when the backend can."""


def digest_text(text: str) -> str:
    """Return ``sha256:<hex>`` for a UTF-8 string."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class ScriptedGrounding:
    """Deterministic boxes keyed by timestamp. Used by fixtures and tests."""

    provider_id = "scripted"
    provenance = None

    def __init__(self, table: dict[float, list[Detection]]):
        self._table = {round(stamp, 6): list(rows) for stamp, rows in table.items()}

    def ground(self, frame: FrameSignal, prompts: list[Prompt]) -> list[Detection]:
        del prompts
        return list(self._table.get(round(frame.t_sec, 6), []))


class ScriptedCount:
    """Deterministic counts keyed by timestamp."""

    provider_id = "scripted_count"

    def __init__(self, table: dict[float, list[CountHit]]):
        self._table = {round(stamp, 6): list(rows) for stamp, rows in table.items()}

    def count(self, frame: FrameSignal, prompts: list[Prompt]) -> list[CountHit]:
        del prompts
        return list(self._table.get(round(frame.t_sec, 6), []))


class KinematicMask:
    """Propagate boxes by the tracker. No neural memory to reset."""

    provider_id = "kinematic"

    def reset(self, reason: str) -> None:
        del reason

    def refine(self, frame: FrameSignal, detections: list[Detection]) -> list[Detection]:
        del frame
        return detections


class UnavailableGrounding:
    """Explicit empty provider when the pinned model cannot be loaded."""

    provider_id = "unavailable"
    provenance = None

    def ground(self, frame: FrameSignal, prompts: list[Prompt]) -> list[Detection]:
        del frame, prompts
        return []


def probe_providers() -> list[ProviderProbe]:
    """Measure each candidate once, then release it.

    Kinematic propagation is always available. CountGD is reported and not
    loaded: ss-perception v0.2.0 does not ship it.
    """
    probes = [
        ProviderProbe(
            provider_id="kinematic",
            role="mask",
            license_name="none",
            license_decision="allowed",
            available=True,
            latency_sec=0.0,
            peak_vram_bytes=0,
            detail="box motion between grounding calls; no neural weights",
        ),
        ProviderProbe(
            provider_id="countgd",
            role="count",
            license_name="unset",
            license_decision="unset",
            available=False,
            detail="not in ss-perception v0.2.0; counts are not identities",
        ),
        _import_probe(
            "yolo_sam",
            "grounding",
            "AGPL-3.0",
            "allowed",
            "ultralytics",
            "YOLO11 is the existing closed-set path",
        ),
        _import_probe(
            "rf_detr",
            "grounding",
            "Apache-2.0",
            "allowed",
            "rfdetr",
            "RF-DETR is the existing tracking path",
        ),
        _import_probe(
            "sam2",
            "mask",
            "Apache-2.0",
            "allowed",
            "sam2",
            "SAM 2 memory is reset on hard cuts",
        ),
        _import_probe(
            "sam3",
            "joint",
            "SAM-License",
            "unset",
            "sam3",
            "SAM 3 is an alternative provider; license is not accepted",
        ),
    ]
    probes.append(_probe_grounding_dino())
    return probes


def _import_probe(
    provider_id: str,
    role: str,
    license_name: str,
    license_decision: str,
    module: str,
    detail: str,
) -> ProviderProbe:
    try:
        importlib.import_module(module)
    except ImportError:
        return ProviderProbe(
            provider_id=provider_id,
            role=role,
            license_name=license_name,
            license_decision=license_decision,
            available=False,
            detail=f"{detail}; import {module} failed",
        )
    return ProviderProbe(
        provider_id=provider_id,
        role=role,
        license_name=license_name,
        license_decision=license_decision,
        available=False,
        detail=f"{detail}; package imports but this benchmark does not load its weights",
    )


def _probe_grounding_dino() -> ProviderProbe:
    """Load Grounding DINO tiny on CUDA when transformers can see a GPU.

    The object-detection pipeline in the pinned transformers release does not
    run ``GroundingDinoForObjectDetection``. This probe calls the zero-shot
    model and processor directly.
    """
    license_name = "Apache-2.0"
    decision = "allowed"
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        return ProviderProbe(
            provider_id="grounding_dino",
            role="grounding",
            license_name=license_name,
            license_decision=decision,
            available=False,
            detail=f"import failed: {exc.__class__.__name__}",
        )
    if not torch.cuda.is_available():
        return ProviderProbe(
            provider_id="grounding_dino",
            role="grounding",
            license_name=license_name,
            license_decision=decision,
            available=False,
            detail="cuda unavailable",
        )
    loaded = None
    try:
        torch.cuda.reset_peak_memory_stats()
        loaded = _load_grounding_dino()
        image = Image.new("RGB", (640, 480), (32, 32, 32))
        prompts = [Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")]
        _detect_grounding_dino(loaded, image, prompts)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _detect_grounding_dino(loaded, image, prompts)
        end.record()
        torch.cuda.synchronize()
        latency = float(start.elapsed_time(end)) / 1000.0
        peak = int(torch.cuda.max_memory_allocated())
        return ProviderProbe(
            provider_id="grounding_dino",
            role="grounding",
            license_name=license_name,
            license_decision=decision,
            available=True,
            latency_sec=latency,
            peak_vram_bytes=peak,
            detail=GROUNDING_DINO_ID,
        )
    except Exception as exc:
        return ProviderProbe(
            provider_id="grounding_dino",
            role="grounding",
            license_name=license_name,
            license_decision=decision,
            available=False,
            detail=f"load failed: {exc.__class__.__name__}: {exc}"[:400],
        )
    finally:
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _prepare_hf_cache() -> None:
    import os
    from pathlib import Path

    from selfsuvis.pipeline.core import settings

    root = Path(settings.data_dir()) / "hf-cache"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))


class GroundingDinoProvider:
    """Text-prompted boxes from IDEA-Research/grounding-dino-tiny.

    The model is loaded on the first ``ground`` call. A load failure sets
    ``failed`` and returns no boxes so the pass can record
    ``provider_unavailable``. A later detection error returns no boxes for
    that frame only.
    """

    provider_id = "grounding_dino"
    model_id = GROUNDING_DINO_ID

    def __init__(self) -> None:
        self._loaded = None
        self.failed = False
        self.provenance = None

    def ground(self, frame: FrameSignal, prompts: list[Prompt]) -> list[Detection]:
        if frame.image is None or self.failed:
            return []
        loaded = self._load()
        if loaded is None:
            return []
        try:
            return _detect_grounding_dino(loaded, frame.image, prompts)
        except Exception:
            return []

    def _load(self):
        if self._loaded is not None or self.failed:
            return self._loaded
        try:
            self._loaded = _load_grounding_dino()
            self.provenance = _provenance(self.model_id, "allowed")
        except Exception:
            self.failed = True
            self._loaded = None
        return self._loaded


def _load_grounding_dino():
    """Return ``(processor, model, device)`` for the pinned Grounding DINO tiny."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    _prepare_hf_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_ID, dtype=dtype)
    model = model.to(device)
    model.eval()
    return processor, model, device, dtype


def _detect_grounding_dino(loaded, image, prompts: list[Prompt]) -> list[Detection]:
    import torch

    processor, model, device, dtype = loaded
    phrases = [prompt.text.strip().lower() for prompt in prompts if prompt.text.strip()]
    if not phrases:
        return []
    text = " ".join(f"{phrase}." for phrase in phrases)
    inputs = processor(images=image, text=text, return_tensors="pt")
    inputs = inputs.to(device)
    if dtype == torch.float16 and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    with torch.no_grad():
        outputs = model(**inputs)
    width, height = image.size
    processed = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.25,
        text_threshold=0.25,
        target_sizes=[(height, width)],
    )
    return _detections_from_grounded(processed[0], prompts, width, height)


def _detections_from_grounded(
    result, prompts: list[Prompt], width: int, height: int
) -> list[Detection]:
    boxes = result.get("boxes")
    scores = result.get("scores")
    labels = result.get("text_labels")
    if labels is None:
        labels = result.get("labels")
    if boxes is None or scores is None:
        return []
    by_text = {prompt.text.strip().lower(): prompt for prompt in prompts}
    detections: list[Detection] = []
    for index, box in enumerate(boxes):
        values = [float(item) for item in box]
        if len(values) != 4 or width <= 0 or height <= 0:
            continue
        x1, y1, x2, y2 = values
        label = ""
        if labels is not None and index < len(labels):
            label = str(labels[index])
        prompt = by_text.get(label.strip().lower().rstrip(".")) or prompts[0]
        detections.append(
            Detection(
                prompt_id=prompt.prompt_id,
                label_raw=label or prompt.text,
                label_normalized=prompt.normalized,
                xywh=(
                    x1 / width,
                    y1 / height,
                    max(1e-3, (x2 - x1) / width),
                    max(1e-3, (y2 - y1) / height),
                ),
                confidence=max(0.0, min(1.0, float(scores[index]))),
                negative_prompts=prompt.negative,
                exemplars=prompt.exemplars,
            )
        )
    return detections


def _provenance(model_id: str, decision: str):
    from selfsuvis.pipeline.analysis4d.schemas import ModelProvenance

    revision, weights = _weights_digest(model_id)
    return ModelProvenance(
        model_id=model_id,
        revision=revision,
        weights_digest=weights,
        license_decision=decision,
        prompt_config_digest=digest_text(_PREPROCESS),
        preprocessing_version=_PREPROCESS,
    )


def _weights_digest(model_id: str) -> tuple[str, str]:
    """Return ``(revision, sha256:...)`` for the cached Grounding DINO weights.

    Hugging Face names each blob with the sha256 of the file. The snapshot
    directory name is the hub revision. When the cache is missing, the digest
    is the sha256 of the model id and the revision is ``main``.
    """
    import os
    from pathlib import Path

    root = Path(os.environ.get("HF_HOME", "")) / "hub"
    folder = root / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if folder.is_dir():
        snapshots = sorted(path for path in folder.iterdir() if path.is_dir())
        if snapshots:
            weight = snapshots[-1] / "model.safetensors"
            if weight.is_symlink():
                blob = Path(os.readlink(weight)).name
                if len(blob) == 64 and all(char in "0123456789abcdef" for char in blob):
                    return snapshots[-1].name, "sha256:" + blob
    return "main", digest_text(model_id + "\nmain")
