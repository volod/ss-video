"""Generate .data/.env.local from environment presets plus resource-aware overrides."""

import argparse
import os
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import selfsuvis
from selfsuvis.pipeline.core.db_urls import sibling_database_url
from selfsuvis.pipeline.core.env import project_roots
from selfsuvis.pipeline.vision.registry import detect_resources

_ENV_NAMES = ("dev", "test", "prod")
_PROFILE_NAMES = ("minimal", "balanced", "full")
_SIDECAR_BACKENDS = ("auto", "none", "ollama", "vllm")
_REASONING_BACKENDS = ("auto", "none", "ollama")
_SECRET_KEYS = ("HF_TOKEN", "API_KEY", "CVAT_API_TOKEN", "CVAT_WEBHOOK_SECRET")


@dataclass(frozen=True)
class ResourceProfile:
    vram_gb: float
    free_vram_gb: float
    ram_gb: float


@dataclass(frozen=True)
class EnvGenerationOptions:
    env_name: str
    profile: str
    output_path: Path
    resources: ResourceProfile
    gemma_backend: str = "auto"
    qwen_backend: str = "auto"
    unidrive_backend: str = "auto"
    reasoning_backend: str = "auto"
    gemma_model: str = ""
    qwen_model: str = ""
    unidrive_model: str = ""
    reasoning_model: str = ""


@dataclass(frozen=True)
class EnvPlan:
    values: dict[str, str]
    resources: ResourceProfile
    profile: str
    env_name: str
    selected_backends: dict[str, str]


def _repo_root() -> Path:
    return project_roots(__file__)[1]


def _parse_env_lines(lines: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_env_template(env_name: str) -> dict[str, str]:
    if env_name not in _ENV_NAMES:
        raise ValueError(f"Unsupported environment: {env_name}")
    names = (f"{env_name}.env",)
    candidates: list[Path] = []
    try:
        candidates.append(Path(str(resources.files("selfsuvis").joinpath("env", names[0]))))
    except (FileNotFoundError, ModuleNotFoundError, TypeError, ValueError):
        pass
    for root in getattr(selfsuvis, "__path__", []):
        candidates.append(Path(root) / "env" / names[0])
    for template in candidates:
        if template.is_file():
            return _parse_env_lines(template.read_text(encoding="utf-8").splitlines())
    raise FileNotFoundError(f"selfsuvis/env/{env_name}.env not found on the namespace path")


def _read_existing_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return _parse_env_lines(path.read_text(encoding="utf-8").splitlines())


def _profile_defaults(profile: str) -> dict[str, str]:
    if profile == "minimal":
        return {
            "gemma_backend": "ollama",
            "qwen_backend": "none",
            "unidrive_backend": "none",
            "reasoning_backend": "ollama",
        }
    if profile == "full":
        return {
            "gemma_backend": "ollama",
            "qwen_backend": "ollama",
            "unidrive_backend": "ollama",
            "reasoning_backend": "ollama",
        }
    return {
        "gemma_backend": "ollama",
        "qwen_backend": "ollama",
        "unidrive_backend": "none",
        "reasoning_backend": "ollama",
    }


def _resolve_backend(kind: str, requested: str, profile: str) -> str:
    if requested != "auto":
        return requested
    return _profile_defaults(profile)[f"{kind}_backend"]


def _recommend_ollama_gemma_models(resources: ResourceProfile) -> tuple[str, str]:
    """Return (gemma_model, reasoning_model) appropriate for the detected hardware.

    Reasoning model tiers:
      ≥ 64 GB free VRAM  → deepseek-r1:32b  (high-end server / multi-GPU)
      ≥ 32 GB free VRAM  → qwen3:30b        (A100 / single H100)
      ≥ 18 GB free VRAM  → deepseek-r1:14b  (RTX 3090/4090 headroom)
      ≥ 14 GB free VRAM  → qwen3:14b        (24 GB card with other models evicted)
      ≥  8 GB free VRAM  → qwen3:14b        (12-16 GB card with one sidecar resident)
      ≥  4 GB free VRAM  → qwen3:8b         (fallback for tighter 8-12 GB systems)
      CPU / low VRAM     → falls back to RAM tiers
    """
    vram = resources.vram_gb
    free_vram = resources.free_vram_gb if resources.free_vram_gb > 0 else vram
    ram = resources.ram_gb

    if free_vram >= 64 or vram >= 80:
        return "gemma4:26b", "deepseek-r1:32b"
    if free_vram >= 32 or vram >= 48:
        return "gemma4:12b", "qwen3:30b"
    if free_vram >= 18 or vram >= 24:
        return "gemma4:4b", "deepseek-r1:14b"
    if free_vram >= 8 or vram >= 14:
        return "gemma4:e4b", "qwen3:14b"
    if free_vram >= 4 or vram >= 8:
        return "gemma4:e4b", "qwen3:8b"
    if ram >= 96:
        return "gemma4:12b", "deepseek-r1:32b"
    if ram >= 64:
        return "gemma4:4b", "qwen3:14b"
    if ram >= 32:
        return "gemma4:e4b", "qwen3:8b"
    return "gemma3:1b", "gemma4:e4b"


def _map_gemma_to_vllm(model_name: str) -> str:
    mapping = {
        "gemma4:e4b": "google/gemma-4-4b-it",
        "gemma4:4b": "google/gemma-4-4b-it",
        "gemma4:12b": "google/gemma-4-12b-it",
        "gemma4:26b": "google/gemma-4-26b-it",
        "gemma4:31b": "google/gemma-4-31b-it",
        "gemma3:1b": "google/gemma-3-1b-it",
        "gemma3:4b": "google/gemma-3-4b-it",
        "gemma3:12b": "google/gemma-3-12b-it",
    }
    return mapping.get(model_name, "google/gemma-4-4b-it")


def _recommend_qwen_model(resources: ResourceProfile, backend: str) -> str:
    vram = resources.vram_gb
    free_vram = resources.free_vram_gb if resources.free_vram_gb > 0 else vram

    if backend == "ollama":
        return "qwen2.5vl:7b" if free_vram >= 10 or vram >= 12 else "qwen2.5vl:3b"
    if free_vram >= 64 or vram >= 80:
        return "Qwen/Qwen2.5-VL-32B-Instruct"
    if free_vram >= 14 or vram >= 16:
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    return "Qwen/Qwen2.5-VL-3B-Instruct"


def _recommend_unidrive_model(resources: ResourceProfile, backend: str) -> str:
    if backend == "ollama":
        return (
            "qwen2.5vl:7b" if resources.ram_gb >= 32 or resources.vram_gb >= 10 else "qwen2.5vl:3b"
        )
    if resources.vram_gb < 8 and resources.ram_gb < 32:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    return "owl10/UniDriveVLA_Nusc_Base_Stage3"


def build_env_plan(
    options: EnvGenerationOptions, existing: Mapping[str, str] | None = None
) -> EnvPlan:
    base = load_env_template(options.env_name)
    existing_values = dict(existing or {})
    gemma_backend = _resolve_backend("gemma", options.gemma_backend, options.profile)
    qwen_backend = _resolve_backend("qwen", options.qwen_backend, options.profile)
    unidrive_backend = _resolve_backend("unidrive", options.unidrive_backend, options.profile)
    reasoning_backend = _resolve_backend("reasoning", options.reasoning_backend, options.profile)

    gemma_ollama_model, reasoning_ollama_model = _recommend_ollama_gemma_models(options.resources)
    gemma_model = options.gemma_model
    if not gemma_model and gemma_backend == "ollama":
        gemma_model = gemma_ollama_model
    elif not gemma_model and gemma_backend == "vllm":
        gemma_model = _map_gemma_to_vllm(gemma_ollama_model)

    reasoning_model = options.reasoning_model
    if not reasoning_model and reasoning_backend == "ollama":
        reasoning_model = reasoning_ollama_model

    qwen_model = options.qwen_model or (
        _recommend_qwen_model(options.resources, qwen_backend) if qwen_backend != "none" else ""
    )
    unidrive_model = options.unidrive_model or (
        _recommend_unidrive_model(options.resources, unidrive_backend)
        if unidrive_backend != "none"
        else ""
    )

    values: dict[str, str] = dict(base)
    values["APP_ENV"] = options.env_name
    values["DEVICE"] = "cpu" if options.resources.vram_gb <= 0 else values.get("DEVICE", "auto")
    values["USE_FP16"] = (
        "false" if options.resources.vram_gb <= 0 else values.get("USE_FP16", "true")
    )
    values["GPU_TOTAL_GB_HINT"] = _format_float(options.resources.vram_gb)
    values["GPU_FREE_GB_HINT"] = _format_float(options.resources.free_vram_gb)
    values.setdefault("HF_TOKEN", "")
    values.setdefault("API_KEY", "")
    values.setdefault(
        "ALLOWED_INDEX_PATHS", os.path.join(values.get("DATA_DIR", "./.data"), "videos")
    )
    values.setdefault("DATABASE_URL", _default_database_url(options.env_name))
    values.setdefault(
        "FUSION_DATABASE_URL",
        sibling_database_url(values["DATABASE_URL"], "selfsuvis_fusion"),
    )
    values.setdefault("CORRELATOR_ENABLED", "true")
    values.setdefault("CORRELATOR_REDIS_URL", "redis://localhost:6379/1")
    values.setdefault("WEBHOOK_REDIS_URL", "redis://localhost:6379/2")
    values.setdefault("STATE_FUSION_ENABLED", "true")
    values.setdefault("MODEL_NAME", "openclip")
    values.setdefault("OPENCLIP_MODEL", "ViT-B-16")
    values.setdefault("OPENCLIP_PRETRAINED", "openai")
    values.setdefault("YOLO_ENABLED", "true")
    values.setdefault("YOLO_SSG_ENABLED", "true")
    values.setdefault("SAM_ENABLED", "true")
    values.setdefault("RFDETR_ENABLED", "true")
    values["SELFSUVIS_USE_GRAPH"] = (
        "1" if options.profile == "full" else values.get("SELFSUVIS_USE_GRAPH", "")
    )

    _apply_sidecar(values, "GEMMA", gemma_backend, gemma_model, default_port=11434)
    _apply_sidecar(values, "QWEN", qwen_backend, qwen_model, default_port=8010)
    _apply_sidecar(values, "REASONING", reasoning_backend, reasoning_model, default_port=11434)

    values["UNIDRIVE_ENABLED"] = "true" if unidrive_backend != "none" else "false"
    if unidrive_backend == "none":
        values["UNIDRIVE_API_URL"] = ""
        values["UNIDRIVE_BACKEND"] = "vllm"
        values["UNIDRIVE_MODEL"] = ""
    else:
        values["UNIDRIVE_BACKEND"] = unidrive_backend
        values["UNIDRIVE_API_URL"] = _default_sidecar_url("UNIDRIVE", unidrive_backend, 8030)
        values["UNIDRIVE_MODEL"] = unidrive_model

    for key in _SECRET_KEYS:
        preserved = existing_values.get(key) or os.getenv(key, "")
        if preserved and not values.get(key):
            values[key] = preserved

    if reasoning_backend == "ollama" and not values.get("REASONING_API_URL"):
        values["REASONING_API_URL"] = values.get("GEMMA_API_URL", "")

    selected_backends = {
        "gemma": gemma_backend,
        "qwen": qwen_backend,
        "unidrive": unidrive_backend,
        "reasoning": reasoning_backend,
    }
    return EnvPlan(
        values=values,
        resources=options.resources,
        profile=options.profile,
        env_name=options.env_name,
        selected_backends=selected_backends,
    )


def _apply_sidecar(
    values: MutableMapping[str, str], prefix: str, backend: str, model: str, *, default_port: int
) -> None:
    model_key = f"{prefix}_MODEL"
    backend_key = f"{prefix}_BACKEND"
    if prefix == "GEMMA":
        model_key = "GEMMA_API_MODEL"
        backend_key = "GEMMA_API_BACKEND"
    if backend == "none":
        values[f"{prefix}_API_URL"] = ""
        if prefix != "REASONING":
            values[backend_key] = "ollama" if prefix == "GEMMA" else "vllm"
        else:
            values["REASONING_BACKEND"] = ""
        values[model_key] = ""
        return
    values[f"{prefix}_API_URL"] = _default_sidecar_url(prefix, backend, default_port)
    if prefix != "REASONING":
        values[backend_key] = backend
    else:
        values["REASONING_BACKEND"] = backend
    values[model_key] = model


def _default_sidecar_url(prefix: str, backend: str, port: int) -> str:
    if backend == "ollama":
        return "http://localhost:11434/v1"
    if prefix == "GEMMA":
        return "http://localhost:8000/v1"
    return f"http://localhost:{port}/v1"


def _default_database_url(env_name: str) -> str:
    if env_name == "prod":
        return "postgresql://selfsuvis:selfsuvis@postgres:5432/selfsuvis"
    return "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis"


def _format_float(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def render_env(plan: EnvPlan) -> str:
    values = dict(plan.values)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines: list[str] = [
        "# Generated by selfsuvis-env",
        f"# Environment: {plan.env_name}",
        f"# Profile: {plan.profile}",
        (
            "# Target resources: "
            f"VRAM(total)={plan.resources.vram_gb:.1f} GiB, "
            f"VRAM(free)={plan.resources.free_vram_gb:.1f} GiB, "
            f"RAM={plan.resources.ram_gb:.1f} GiB"
        ),
        (
            "# Sidecars: "
            f"gemma={plan.selected_backends['gemma']}, "
            f"qwen={plan.selected_backends['qwen']}, "
            f"unidrive={plan.selected_backends['unidrive']}, "
            f"reasoning={plan.selected_backends['reasoning']}"
        ),
        f"# Generated at: {generated_at}",
        "",
        "# Secrets",
        f"HF_TOKEN={values.pop('HF_TOKEN', '')}",
        f"API_KEY={values.pop('API_KEY', '')}",
        f"CVAT_API_TOKEN={values.pop('CVAT_API_TOKEN', '')}",
        f"CVAT_WEBHOOK_SECRET={values.pop('CVAT_WEBHOOK_SECRET', '')}",
        "",
    ]

    ordered_groups: Sequence[tuple[str, Sequence[str]]] = (
        (
            "ss-video",
            (
                "APP_ENV",
                "DATA_DIR",
                "FRAMES_DIR",
                "TILES_DIR",
                "VIDEOS_DIR",
                "REPORTS_DIR",
                "MAPS_DIR",
                "DATABASE_URL",
                "QDRANT_HOST",
                "QDRANT_PORT",
                "QDRANT_COLLECTION",
                "DEVICE",
                "USE_FP16",
                "GPU_TOTAL_GB_HINT",
                "GPU_FREE_GB_HINT",
                "ALLOWED_INDEX_PATHS",
                "LOG_LEVEL",
            ),
        ),
        (
            "ss-fusion",
            (
                "FUSION_DATABASE_URL",
                "CORRELATOR_ENABLED",
                "CORRELATOR_REDIS_URL",
                "WEBHOOK_REDIS_URL",
                "WEBHOOK_ALERT_URL",
                "STATE_FUSION_ENABLED",
            ),
        ),
        (
            "models",
            (
                "MODEL_NAME",
                "OPENCLIP_MODEL",
                "OPENCLIP_PRETRAINED",
                "GEMMA_MODEL_ID",
            ),
        ),
        (
            "sidecars",
            (
                "GEMMA_API_URL",
                "GEMMA_API_BACKEND",
                "GEMMA_API_MODEL",
                "QWEN_API_URL",
                "QWEN_BACKEND",
                "QWEN_MODEL",
                "REASONING_API_URL",
                "REASONING_BACKEND",
                "REASONING_MODEL",
                "UNIDRIVE_ENABLED",
                "UNIDRIVE_API_URL",
                "UNIDRIVE_BACKEND",
                "UNIDRIVE_MODEL",
                "FLORENCE_API_URL",
                "FLORENCE_MODEL",
            ),
        ),
        (
            "pipeline",
            (
                "YOLO_ENABLED",
                "YOLO_MODEL",
                "YOLO_SSG_ENABLED",
                "SAM_ENABLED",
                "SAM_MODEL",
                "RFDETR_ENABLED",
                "RFDETR_MODEL",
                "SELFSUVIS_USE_GRAPH",
                "SAMPLE_FPS_BASE",
                "SAMPLE_FPS_MIN",
                "SAMPLE_FPS_MAX",
                "HIST_THRESH",
                "EMBED_DRIFT_THRESH",
                "MAX_GAP_SEC",
                "TILE_SIZE",
                "STRIDE",
                "MAX_TILES_PER_SEGMENT",
                "DEDUP_COS_SIM_THRESH",
            ),
        ),
        (
            "services",
            (
                "STATIC_SERVER_URL",
                "SUPERSPLAT_SERVER_URL",
                "NERFSTUDIO_API_URL",
                "MAPPER_API_URL",
            ),
        ),
    )

    group_headings = (
        "ss-video",
        "ss-fusion",
        "Model selection",
        "Inference sidecars",
        "Pipeline flags",
        "Service endpoints",
    )
    for index, (_group_name, keys) in enumerate(ordered_groups):
        if index > 0:
            lines.append("")
        lines.append(f"# {group_headings[index]}")
        for key in keys:
            if key in values:
                lines.append(f"{key}={values.pop(key)}")

    if values:
        lines.append("")
        lines.append("# Additional template values")
        for key in sorted(values):
            lines.append(f"{key}={values[key]}")

    lines.append("")
    return "\n".join(lines)


def write_env_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _prompt_choice(prompt: str, options: Sequence[str], default: str) -> str:
    options_label = "/".join(options)
    while True:
        raw = input(f"{prompt} [{default}] ({options_label}): ").strip()
        if not raw:
            return default
        if raw in options:
            return raw
        print(f"Expected one of: {', '.join(options)}", file=sys.stderr)


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default:.1f}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Enter a numeric value.", file=sys.stderr)


def _prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def _interactive_options(args: argparse.Namespace, detected: ResourceProfile) -> argparse.Namespace:
    if not sys.stdin.isatty():
        raise RuntimeError("--interactive requires a TTY")

    # -- 1. Show detected hardware --------------------------------------------
    gpu_str = (
        f"{detected.vram_gb:.1f} GiB GPU ({detected.free_vram_gb:.1f} GiB free)"
        if detected.vram_gb > 0
        else "CPU only (no GPU detected)"
    )
    print(f"\n  Hardware: {gpu_str},  {detected.ram_gb:.1f} GiB RAM")

    rec_gemma, rec_reasoning = _recommend_ollama_gemma_models(detected)
    print(f"  Recommended models (Ollama): Gemma={rec_gemma}  Reasoning={rec_reasoning}\n")

    # -- 2. Primary sidecar backend -------------------------------------------
    print("  Sidecar backends:")
    print("    ollama  — local server, models pulled automatically, easiest setup")
    print("    vllm    — higher throughput for batch inference, GPU recommended")
    print("    none    — embedding + captioning only (no generative sidecars)")
    primary = _prompt_choice("Primary sidecar backend", ["ollama", "vllm", "none"], "ollama")

    # Apply primary to generative sidecars.  Reasoning always uses Ollama
    # (it runs small reasoning models that vLLM adds little benefit for).
    if primary == "none":
        args.gemma_backend = args.qwen_backend = args.unidrive_backend = args.reasoning_backend = (
            "none"
        )
    else:
        args.gemma_backend = primary
        args.qwen_backend = primary
        args.unidrive_backend = "none"  # UniDrive is off by default; enable explicitly
        args.reasoning_backend = "ollama"  # reasoning stays on Ollama regardless

    # -- 3. Profile -----------------------------------------------------------
    print("\n  Profiles:")
    print("    minimal  — Gemma + Reasoning only (fastest, lowest resource use)")
    print("    balanced — Gemma + Qwen + Reasoning (good default)")
    print("    full     — all sidecars including UniDrive")
    args.profile = _prompt_choice("Profile", _PROFILE_NAMES, "balanced")

    # -- 4. Reasoning model ---------------------------------------------------
    if args.reasoning_backend != "none":
        _reasoning_options = {
            "qwen3:8b": "~5 GB  — fast fallback for tighter 8-12 GB systems",
            "qwen3:14b": "~8 GB  — recommended for 12 GB+ cards when Ollama keeps one model loaded",
            "deepseek-r1:14b": "~9 GB  — R1-distilled reasoning specialist, similar size",
            "qwen3:30b": "~18 GB — high quality, needs 24+ GB free VRAM",
            "deepseek-r1:32b": "~19 GB — best reasoning quality, needs 32+ GB free VRAM",
        }
        print("\n  Reasoning model (step 24 — agentic flow audit):")
        print(f"  Detected hardware recommendation: {rec_reasoning}")
        print("  Available options:")
        for tag, note in _reasoning_options.items():
            marker = " ◀ recommended" if tag == rec_reasoning else ""
            print(f"    {tag:<26}  {note}{marker}")
        args.reasoning_model = _prompt_text("Ollama reasoning model tag", rec_reasoning)

    # -- 5. Environment and output --------------------------------------------
    args.env_name = _prompt_choice("Environment", _ENV_NAMES, args.env_name)
    args.output = _prompt_text("Output path", args.output)
    return args


def _resource_profile_from_args(args: argparse.Namespace) -> ResourceProfile:
    # Strip GPU hint overrides that a previous generation may have written into
    # the .env so that detect_resources() always calls nvidia-smi fresh.
    os.environ.pop("GPU_TOTAL_GB_HINT", None)
    os.environ.pop("GPU_FREE_GB_HINT", None)
    detected = detect_resources()
    vram = args.vram_gb if args.vram_gb is not None else float(detected.get("vram_gb", 0.0))
    free_vram = (
        args.free_vram_gb
        if args.free_vram_gb is not None
        else float(detected.get("free_vram_gb", vram))
    )
    ram = args.ram_gb if args.ram_gb is not None else float(detected.get("ram_gb", 8.0))
    return ResourceProfile(vram_gb=vram, free_vram_gb=free_vram, ram_gb=ram)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate .data/.env.local from packaged env presets and resource-aware defaults.",
    )
    parser.add_argument(
        "--env",
        dest="env_name",
        choices=_ENV_NAMES,
        default="dev",
        help="Base environment preset to start from.",
    )
    parser.add_argument(
        "--profile", choices=_PROFILE_NAMES, default="balanced", help="High-level sidecar profile."
    )
    parser.add_argument(
        "--output", default=".data/.env.local", help="Output path for the generated env file."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for environment, resources, and sidecar choices.",
    )
    parser.add_argument(
        "--vram-gb",
        type=float,
        default=None,
        help="Target total GPU VRAM in GiB. Defaults to detected resources.",
    )
    parser.add_argument(
        "--free-vram-gb",
        type=float,
        default=None,
        help="Target free GPU VRAM in GiB. Defaults to detected resources.",
    )
    parser.add_argument(
        "--ram-gb",
        type=float,
        default=None,
        help="Target system RAM in GiB. Defaults to detected resources.",
    )
    parser.add_argument("--gemma-backend", choices=_SIDECAR_BACKENDS, default="auto")
    parser.add_argument("--qwen-backend", choices=_SIDECAR_BACKENDS, default="auto")
    parser.add_argument("--unidrive-backend", choices=_SIDECAR_BACKENDS, default="auto")
    parser.add_argument("--reasoning-backend", choices=_REASONING_BACKENDS, default="auto")
    parser.add_argument(
        "--gemma-model", default="", help="Override the generated Gemma sidecar model."
    )
    parser.add_argument(
        "--qwen-model", default="", help="Override the generated Qwen sidecar model."
    )
    parser.add_argument(
        "--unidrive-model", default="", help="Override the generated UniDrive sidecar model."
    )
    parser.add_argument(
        "--reasoning-model", default="", help="Override the generated reasoning model."
    )
    return parser


def _print_sidecar_next_steps(plan: EnvPlan, values: dict[str, str]) -> None:
    """Print the sidecar startup commands implied by the generated plan."""
    backends = plan.selected_backends
    ollama_models: list[str] = []
    vllm_cmds: list[str] = []

    gemma_model = values.get("GEMMA_API_MODEL", "")
    qwen_model = values.get("QWEN_MODEL", "")
    reasoning_model = values.get("REASONING_MODEL", "")
    unidrive_model = values.get("UNIDRIVE_MODEL", "")

    if backends.get("gemma") == "ollama" and gemma_model:
        ollama_models.append(gemma_model)
    elif backends.get("gemma") == "vllm" and gemma_model:
        vllm_cmds.append(
            f"python -m vllm.entrypoints.openai.api_server \\\n"
            f"  --model {gemma_model} --port 8000 --max-model-len 8192"
        )

    if backends.get("qwen") == "ollama" and qwen_model:
        ollama_models.append(qwen_model)
    elif backends.get("qwen") == "vllm" and qwen_model:
        vllm_cmds.append(
            f"python -m vllm.entrypoints.openai.api_server \\\n"
            f"  --model {qwen_model} --port 8010 --max-model-len 8192"
        )

    if backends.get("reasoning") == "ollama" and reasoning_model:
        ollama_models.append(reasoning_model)

    if backends.get("unidrive") == "vllm" and unidrive_model:
        vllm_cmds.append(
            f"python -m vllm.entrypoints.openai.api_server \\\n"
            f"  --model {unidrive_model} --port 8030 --max-model-len 4096"
        )

    if not ollama_models and not vllm_cmds:
        return

    print("\n--- Sidecar startup commands --------------------------------------")
    if ollama_models:
        print("\n  Ollama (run once, then keep running):")
        print("    ollama serve")
        for m in dict.fromkeys(ollama_models):  # deduplicate, preserve order
            print(f"    ollama pull {m}")

    if vllm_cmds:
        print("\n  vLLM (each in its own terminal):")
        for cmd in vllm_cmds:
            print(f"    {cmd}\n")

    print("\n  Edit .data/.env.local and set API_KEY, then drop videos into .data/videos/ and run:")
    print("    make up  (Docker)  or  make venv && uvicorn selfsuvis.app.main:app  (local)")
    print("------------------------------------------------------------------\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    detected = _resource_profile_from_args(args)
    if args.interactive:
        args = _interactive_options(args, detected)
    resources_profile = _resource_profile_from_args(args)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _repo_root() / output_path

    options = EnvGenerationOptions(
        env_name=args.env_name,
        profile=args.profile,
        output_path=output_path,
        resources=resources_profile,
        gemma_backend=args.gemma_backend,
        qwen_backend=args.qwen_backend,
        unidrive_backend=args.unidrive_backend,
        reasoning_backend=args.reasoning_backend,
        gemma_model=args.gemma_model,
        qwen_model=args.qwen_model,
        unidrive_model=args.unidrive_model,
        reasoning_model=args.reasoning_model,
    )
    existing = _read_existing_env(output_path)
    plan = build_env_plan(options, existing=existing)
    contents = render_env(plan)
    write_env_file(output_path, contents)

    print(
        f"\nWrote {output_path}\n"
        f"  env={plan.env_name}  profile={plan.profile}\n"
        f"  gemma={plan.selected_backends['gemma']}  "
        f"qwen={plan.selected_backends['qwen']}  "
        f"reasoning={plan.selected_backends['reasoning']}  "
        f"unidrive={plan.selected_backends['unidrive']}\n"
        f"  GPU={plan.resources.vram_gb:.1f} GiB  RAM={plan.resources.ram_gb:.1f} GiB"
    )
    _print_sidecar_next_steps(plan, plan.values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
