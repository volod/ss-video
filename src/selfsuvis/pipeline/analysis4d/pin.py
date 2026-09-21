"""Pinned 4D providers and the gate that selects them.

Production loads one grounding provider and one mask propagator. CountGD is
never pinned as an identity source. A provider that misses the license,
memory, or latency gate is left out of the pin.
"""

import os
from dataclasses import dataclass

from selfsuvis.pipeline.analysis4d.keyframes import SelectorConfig

VRAM_BUDGET_BYTES = 8 * 1024 * 1024 * 1024
LATENCY_BUDGET_SEC = 1.0
GROUNDING_PREFERENCE = ("grounding_dino", "sam3", "rf_detr", "yolo_sam")
MASK_PREFERENCE = ("sam2", "kinematic")

# Reference-host pin. The track benchmark on the CUDA host is what updates this.
PINNED_GROUNDING = "grounding_dino"
PINNED_MASK = "kinematic"
PINNED_COUNT = ""
PINNED_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PINNED_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
PINNED_WEIGHTS_DIGEST = "sha256:1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
PINNED_RELATIVE_DEPTH = "depth-anything/Depth-Anything-V2-Small-hf"
PINNED_METRIC_DEPTH = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
PINNED_APPEARANCE = "dinov3_vitb14"
# Reference-host cache identities. The geometry benchmark checks these when CUDA is probed.
PINNED_RELATIVE_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
PINNED_RELATIVE_WEIGHTS = "sha256:3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
PINNED_METRIC_REVISION = "fd2c22027eaf20374204f14099b8341e1925ad39"
PINNED_METRIC_WEIGHTS = "sha256:ad065c77a7421ca55159a1f0db9433397a607690f2d76bb8a6fc54b1be7a3124"
PINNED_APPEARANCE_REVISION = (
    "sha256:73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71"
)


@dataclass(frozen=True)
class ProviderProbe:
    """One candidate measured on this machine, or marked unavailable."""

    provider_id: str
    role: str
    license_name: str
    license_decision: str
    available: bool
    latency_sec: float | None = None
    peak_vram_bytes: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class PinChoice:
    """The single production pair. ``count`` is empty unless a future pin says otherwise."""

    grounding: str
    mask: str
    count: str
    model_gate: str
    detail: str


def meets_gate(probe: ProviderProbe) -> bool:
    """True when the probe is allowed, loaded, and inside the memory and latency budgets."""
    if not probe.available or probe.license_decision != "allowed":
        return False
    if probe.peak_vram_bytes is not None and probe.peak_vram_bytes > _vram_budget():
        return False
    if probe.latency_sec is not None and probe.latency_sec > _latency_budget():
        return False
    return True


def choose_pin(probes: list[ProviderProbe]) -> PinChoice:
    """Pick one grounding provider and one mask propagator.

    SAM 3 is an alternative to Grounding DINO plus SAM 2, not an extra pass.
    Count probes are recorded by the caller and are not selected here.
    """
    catalog = list(probes)
    if not any(item.provider_id == "kinematic" for item in catalog):
        catalog.append(_kinematic_probe())
    by_id = {item.provider_id: item for item in catalog}
    grounding = ""
    for name in GROUNDING_PREFERENCE:
        probe = by_id.get(name)
        if probe is not None and probe.role in {"grounding", "joint"} and meets_gate(probe):
            grounding = name
            break
    if not grounding:
        return PinChoice(
            grounding="",
            mask="",
            count="",
            model_gate="no-go",
            detail="no grounding provider met license, memory, and latency gates",
        )
    if grounding == "sam3":
        return PinChoice(
            grounding="sam3",
            mask="sam3",
            count="",
            model_gate="pass",
            detail="sam3",
        )
    mask = "kinematic"
    for name in MASK_PREFERENCE:
        probe = by_id.get(name)
        if probe is not None and probe.role == "mask" and meets_gate(probe):
            mask = name
            break
    return PinChoice(
        grounding=grounding,
        mask=mask,
        count="",
        model_gate="pass",
        detail=f"{grounding}+{mask}",
    )


def active_grounding() -> str:
    """Return the configured grounding provider, defaulting to the pin."""
    return os.environ.get("ANALYSIS4D_GROUNDING_PROVIDER", PINNED_GROUNDING).strip()


def active_mask() -> str:
    """Return the configured mask propagator, defaulting to the pin."""
    return os.environ.get("ANALYSIS4D_MASK_PROVIDER", PINNED_MASK).strip()


def active_count() -> str:
    """Return the configured count provider. The default is off."""
    return os.environ.get("ANALYSIS4D_COUNT_PROVIDER", PINNED_COUNT).strip()


def active_seed() -> int:
    """Return the configured deterministic seed."""
    return _env_int("ANALYSIS4D_SEED", 0)


def selector_config() -> SelectorConfig:
    """Build the selector config from the environment and the pinned seed."""
    return SelectorConfig(
        seed=active_seed(),
        max_gap_sec=_env_float("ANALYSIS4D_MAX_GAP_SEC", 10.0),
        chunk_sec=_env_float("ANALYSIS4D_CHUNK_SEC", 4.0),
        keyframe_budget=max(1, _env_int("ANALYSIS4D_KEYFRAME_BUDGET", 2)),
        min_spacing_sec=_env_float("ANALYSIS4D_MIN_SPACING_SEC", 0.25),
    )


def _vram_budget() -> int:
    return _env_int("ANALYSIS4D_VRAM_BUDGET_BYTES", VRAM_BUDGET_BYTES) or VRAM_BUDGET_BYTES


def _latency_budget() -> float:
    return _env_float("ANALYSIS4D_KEYFRAME_LATENCY_SEC", LATENCY_BUDGET_SEC)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _kinematic_probe() -> ProviderProbe:
    return ProviderProbe(
        provider_id="kinematic",
        role="mask",
        license_name="none",
        license_decision="allowed",
        available=True,
        latency_sec=0.0,
        peak_vram_bytes=0,
        detail="box motion between grounding calls; no neural weights",
    )
