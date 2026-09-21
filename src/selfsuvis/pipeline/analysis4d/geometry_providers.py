"""Depth, calibration, and masked-appearance providers for 4D geometry.

Neural adapters load lazily. Metric3D and Perspective Fields are probed and
left unavailable when this host does not have them. A relative depth map is
never relabeled metric inside this module.
"""

import logging
from dataclasses import dataclass

import numpy as np

from selfsuvis.pipeline.analysis4d.appearance import (
    PREPROCESS,
    Descriptor,
    EmbeddingSpace,
    pool_masked,
)
from selfsuvis.pipeline.analysis4d.camera import CameraModel, PerspectiveEstimate

logger = logging.getLogger(__name__)

RELATIVE_DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"
METRIC_DEPTH_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
ZOEDEPTH_ID = "Intel/zoedepth-nyu-kitti"
APPEARANCE_MODEL = "dinov3_vitb14"
APPEARANCE_CHECKPOINT = "dinov2_vitb14_reg4_pretrain.pth"
_PREPROCESS = "analysis4d-geometry-v1"


@dataclass(frozen=True)
class DepthMap:
    """One depth image. ``scale`` is the provider claim before calibration gating."""

    values: np.ndarray
    scale: str
    normals: np.ndarray | None = None


@dataclass(frozen=True)
class GeometryView:
    """One timestamp the reconstructor can back-project."""

    t_sec: float
    camera: CameraModel | None = None
    image: object | None = None
    mask: np.ndarray | None = None
    feature_map: np.ndarray | None = None


class ScriptedDepth:
    """Return one depth map for every view. Used by fixtures."""

    provider_id = "scripted_depth"

    def __init__(
        self,
        values: np.ndarray,
        scale: str,
        normals: np.ndarray | None = None,
    ):
        self._values = np.asarray(values, dtype=np.float64)
        self.scale = scale
        self._normals = None if normals is None else np.asarray(normals, dtype=np.float64)
        self.failed = False

    def estimate(self, view: GeometryView) -> DepthMap | None:
        del view
        if self.failed:
            return None
        return DepthMap(self._values, self.scale, self._normals)


class FailingDepth:
    """A depth provider that did not load. Callers keep the 2D tracks."""

    provider_id = "unavailable"
    scale = "unavailable"
    failed = True

    def estimate(self, view: GeometryView) -> DepthMap | None:
        del view
        raise RuntimeError("depth provider unavailable")


class ScriptedFields:
    """Fixed roll, pitch, and field of view. Not a metric calibration."""

    provider_id = "scripted_fields"
    failed = False

    def __init__(self, estimate: PerspectiveEstimate):
        self._estimate = estimate

    def estimate(self, view: GeometryView) -> PerspectiveEstimate | None:
        del view
        return self._estimate


class ScriptedAppearance:
    """Fixed descriptor, or a masked pool when the view carries a feature map."""

    provider_id = "scripted_appearance"
    failed = False

    def __init__(self, space: EmbeddingSpace, vector: np.ndarray):
        self.space = space
        self._vector = np.asarray(vector, dtype=np.float64)

    def describe(self, view: GeometryView, mask: np.ndarray) -> Descriptor | None:
        if view.feature_map is not None:
            vector = pool_masked(view.feature_map, mask)
            return Descriptor(self.space, vector, t_sec=view.t_sec)
        return Descriptor(self.space, self._vector, t_sec=view.t_sec)


class HuggingFaceDepth:
    """Relative or metric depth from a transformers depth-estimation model.

    The ``scale`` argument is the model family. This class does not inspect the
    numeric range and promote a relative map to meters.
    """

    def __init__(self, model_id: str, scale: str, provider_id: str):
        self.model_id = model_id
        self.scale = scale
        self.provider_id = provider_id
        self.failed = False
        self.last_error = ""
        self._loaded = None

    def estimate(self, view: GeometryView) -> DepthMap | None:
        if self.failed or view.image is None:
            return None
        loaded = self._load()
        if loaded is None:
            return None
        try:
            values = _predict_depth(loaded, view.image, metric=self.scale == "metric")
        except Exception as exc:
            self.last_error = _hub_failure(self.model_id, exc)
            logger.warning("depth provider %s: %s", self.provider_id, self.last_error)
            return None
        if values is None:
            return None
        return DepthMap(values, self.scale, None)

    def _load(self):
        if self._loaded is not None or self.failed:
            return self._loaded
        try:
            self._loaded = _load_hf_depth(self.model_id)
        except Exception as exc:
            self.failed = True
            self.last_error = _hub_failure(self.model_id, exc)
            logger.warning("depth provider %s: %s", self.provider_id, self.last_error)
            self._loaded = None
        return self._loaded


class DinoAppearance:
    """Masked pool of the project's DINOv3 alias (DINOv2 with register tokens)."""

    provider_id = "dinov3"
    failed = False

    def __init__(self, model_name: str = APPEARANCE_MODEL):
        self.model_name = model_name
        self._loaded = None
        self.space: EmbeddingSpace | None = None

    def describe(self, view: GeometryView, mask: np.ndarray) -> Descriptor | None:
        if self.failed or view.image is None:
            return None
        loaded = self._load()
        if loaded is None or self.space is None:
            return None
        try:
            features = _dino_tokens(*loaded, view.image)
        except Exception:
            return None
        vector = pool_masked(features, mask)
        return Descriptor(self.space, vector, t_sec=view.t_sec)

    def _load(self):
        if self._loaded is not None or self.failed:
            return self._loaded
        try:
            model, device, _revision, digest = _load_dino(self.model_name)
            dim = int(getattr(model, "embed_dim", 0) or 0)
            if dim <= 0:
                dim = int(_dino_tokens(model, device, _blank_image()).shape[-1])
            self.space = EmbeddingSpace(
                model_id=self.model_name,
                revision=digest,
                dim=dim,
                preprocessing_version=PREPROCESS,
            )
            self._loaded = (model, device)
        except Exception:
            self.failed = True
            self._loaded = None
        return self._loaded


def probe_geometry_models() -> list:
    """Measure depth, appearance, and calibration candidates on this machine."""
    from selfsuvis.pipeline.analysis4d.pin import ProviderProbe

    probes = [
        _probe_depth(RELATIVE_DEPTH_ID, "relative", "depth_anything", "Apache-2.0"),
        _probe_depth(METRIC_DEPTH_ID, "metric", "depth_anything_metric", "Apache-2.0"),
        _probe_depth(ZOEDEPTH_ID, "metric", "zoedepth", "Apache-2.0"),
        _probe_metric3d(),
        _probe_dino(),
        _probe_perspective_fields(),
    ]
    return [item if isinstance(item, ProviderProbe) else item for item in probes]


def _probe_depth(model_id: str, scale: str, provider_id: str, license_name: str):
    from selfsuvis.pipeline.analysis4d.pin import ProviderProbe

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        return ProviderProbe(
            provider_id=provider_id,
            role=f"depth_{scale}",
            license_name=license_name,
            license_decision="allowed",
            available=False,
            detail=f"import failed: {exc.__class__.__name__}",
        )
    if not torch.cuda.is_available():
        return ProviderProbe(
            provider_id=provider_id,
            role=f"depth_{scale}",
            license_name=license_name,
            license_decision="allowed",
            available=False,
            detail="cuda unavailable",
        )
    provider = HuggingFaceDepth(model_id, scale, provider_id)
    image = Image.new("RGB", (160, 120), (40, 80, 120))
    view = GeometryView(t_sec=0.0, image=image)
    try:
        torch.cuda.reset_peak_memory_stats()
        warmed = provider.estimate(view)
        if warmed is None or provider.failed:
            raise RuntimeError(provider.last_error or "empty depth")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        predicted = provider.estimate(view)
        end.record()
        torch.cuda.synchronize()
        latency = float(start.elapsed_time(end)) / 1000.0
        peak = int(torch.cuda.max_memory_allocated())
        revision, digest = _hf_weights_digest(model_id)
        finite = predicted is not None and bool(np.isfinite(predicted.values).any())
        return ProviderProbe(
            provider_id=provider_id,
            role=f"depth_{scale}",
            license_name=license_name,
            license_decision="allowed",
            available=finite,
            latency_sec=latency,
            peak_vram_bytes=peak,
            detail=f"{model_id} {revision} {digest} scale={scale}",
        )
    except Exception as exc:
        detail = provider.last_error or f"load failed: {type(exc).__name__}: {exc}"
        return ProviderProbe(
            provider_id=provider_id,
            role=f"depth_{scale}",
            license_name=license_name,
            license_decision="allowed",
            available=False,
            detail=detail[:400],
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _probe_metric3d():
    from selfsuvis.pipeline.analysis4d.pin import ProviderProbe

    try:
        import importlib

        importlib.import_module("metric3d")
    except ImportError:
        return ProviderProbe(
            provider_id="metric3d",
            role="depth_metric",
            license_name="unset",
            license_decision="unset",
            available=False,
            detail="metric3d is not installed in ss-perception v0.2.0; normals come from depth gradients",
        )
    return ProviderProbe(
        provider_id="metric3d",
        role="depth_metric",
        license_name="unset",
        license_decision="unset",
        available=False,
        detail="metric3d imports but no weights are pinned",
    )


def _probe_perspective_fields():
    from selfsuvis.pipeline.analysis4d.pin import ProviderProbe

    try:
        import importlib

        importlib.import_module("perspective_fields")
        available = True
        detail = "perspective_fields imports"
    except ImportError:
        available = False
        detail = "perspective_fields is not installed; metadata and the scripted fallback remain"
    return ProviderProbe(
        provider_id="perspective_fields",
        role="calibration",
        license_name="unset",
        license_decision="unset",
        available=available,
        detail=detail,
    )


def _probe_dino():
    from selfsuvis.pipeline.analysis4d.pin import ProviderProbe

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        return _dino_unavailable(f"import failed: {exc.__class__.__name__}")
    if not torch.cuda.is_available():
        return _dino_unavailable("cuda unavailable")
    appearance = DinoAppearance()
    image = Image.new("RGB", (224, 224), (20, 20, 20))
    mask = np.zeros((224, 224), dtype=bool)
    mask[40:180, 40:180] = True
    view = GeometryView(t_sec=0.0, image=image)
    try:
        torch.cuda.reset_peak_memory_stats()
        warmed = appearance.describe(view, mask)
        if warmed is None or appearance.space is None:
            raise RuntimeError("empty descriptor")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        described = appearance.describe(view, mask)
        end.record()
        torch.cuda.synchronize()
        latency = float(start.elapsed_time(end)) / 1000.0
        peak = int(torch.cuda.max_memory_allocated())
        space = appearance.space
        return ProviderProbe(
            provider_id="dinov3",
            role="appearance",
            license_name="Apache-2.0",
            license_decision="allowed",
            available=described is not None and described.vector.shape == (space.dim,),
            latency_sec=latency,
            peak_vram_bytes=peak,
            detail=f"{space.model_id} {space.revision} dim={space.dim}",
        )
    except Exception as exc:
        return _dino_unavailable(f"load failed: {exc.__class__.__name__}: {exc}"[:400])
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _dino_unavailable(detail: str):
    from selfsuvis.pipeline.analysis4d.pin import ProviderProbe

    return ProviderProbe(
        provider_id="dinov3",
        role="appearance",
        license_name="Apache-2.0",
        license_decision="allowed",
        available=False,
        detail=detail,
    )


def _hub_failure(model_id: str, exc: BaseException) -> str:
    from selfsuvis.pipeline.analysis4d.providers import explain_hub_failure

    return explain_hub_failure(model_id, exc)


def _load_hf_depth(model_id: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    from selfsuvis.pipeline.analysis4d.providers import _prepare_hf_cache, huggingface_token

    _prepare_hf_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    token = huggingface_token()
    auth = {"token": token} if token else {}
    processor = AutoImageProcessor.from_pretrained(model_id, **auth)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=dtype, **auth)
    model = model.to(device)
    model.eval()
    return processor, model, device, dtype


def _predict_depth(loaded, image, *, metric: bool) -> np.ndarray | None:
    import torch

    processor, model, device, dtype = loaded
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    if dtype == torch.float16 and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted = outputs.predicted_depth
    if predicted.ndim == 4:
        predicted = predicted[:, 0]
    width, height = image.size
    predicted = torch.nn.functional.interpolate(
        predicted.unsqueeze(1).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    values = predicted.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(values).any():
        return None
    if metric:
        if float(np.nanmedian(values)) < 0:
            values = -values
        return np.where(values > 1e-6, values, np.nan)
    finite = values[np.isfinite(values)]
    shifted = values - float(finite.min())
    return np.where(np.isfinite(values), shifted + 1e-3, np.nan)


def _load_dino(model_name: str):
    import torch

    from selfsuvis.models.dino_model import hub_load_dino

    model = hub_load_dino(model_name, pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    revision, _digest = _checkpoint_digest(APPEARANCE_CHECKPOINT)
    return model, device, revision, _digest


def _dino_tokens(model, device: str, image) -> np.ndarray:
    import torch
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(tensor)
    tokens = features["x_norm_patchtokens"][0]
    side = int(round(tokens.shape[0] ** 0.5))
    grid = tokens.reshape(side, side, tokens.shape[-1])
    return grid.detach().cpu().numpy().astype(np.float64)


def _blank_image():
    from PIL import Image

    return Image.new("RGB", (224, 224), (0, 0, 0))


def _hf_weights_digest(model_id: str) -> tuple[str, str]:
    """Return ``(revision, sha256:...)`` from whichever hub cache holds the snapshot."""
    import hashlib
    import os
    from pathlib import Path

    from selfsuvis.pipeline.analysis4d.providers import digest_text
    from selfsuvis.pipeline.core import settings

    roots: list[Path] = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        raw = os.environ.get(key, "").strip()
        if raw:
            roots.append(Path(raw))
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        roots.append(Path(home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path(settings.data_dir()) / "hf-cache" / "hub")
    folder_name = "models--" + model_id.replace("/", "--")
    seen: set[Path] = set()
    for root in roots:
        folder = (root / folder_name / "snapshots").resolve()
        if folder in seen or not folder.is_dir():
            continue
        seen.add(folder)
        snapshots = sorted(path for path in folder.iterdir() if path.is_dir())
        if not snapshots:
            continue
        weight = snapshots[-1] / "model.safetensors"
        if not weight.exists():
            continue
        if weight.is_symlink():
            blob = Path(os.readlink(weight)).name
        else:
            blob = hashlib.sha256(weight.read_bytes()).hexdigest()
        if len(blob) == 64 and all(char in "0123456789abcdef" for char in blob):
            return snapshots[-1].name, "sha256:" + blob
    return "main", digest_text(model_id + "\nmain")


def _checkpoint_digest(filename: str) -> tuple[str, str]:
    import hashlib
    from pathlib import Path

    import torch

    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        return "missing", "sha256:" + "0" * 64
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.name, "sha256:" + digest
