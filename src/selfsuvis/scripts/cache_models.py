"""Cache the model weights a default ``make run-full`` preflight requires.

Already cached artifacts are not downloaded again. The command list comes from
the local preflight checks, so the set matches ``ssv --mode local``.
"""

import shlex
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

_MODEL_BINARIES = {"ssv-models", "selfsuvis-models"}


def local_pipeline_args() -> Namespace:
    """Return the local-mode defaults ``make run-full`` uses."""
    return Namespace(
        mode="local",
        output_dir=".data/local_runs",
        videos_dir=None,
        video=None,
        device="auto",
        fps=2.0,
        asr=None,
        ocr=None,
        depth=None,
        detection=None,
        world_model=None,
        qwen=None,
        unidrive=None,
        scenetok=None,
        asr_model="auto",
        asr_language="",
        ocr_model="auto",
        depth_model="auto",
        detection_model="auto",
        detection_labels="",
        world_model_id="auto",
        yolo_model="yolo11l",
        sam_model="auto",
        no_yolo=False,
        no_sam=False,
        no_rfdetr=False,
        rfdetr_model="base",
    )


def pending_cache_commands(required: list[str], missing: list[str]) -> tuple[list[str], list[str]]:
    """Split required cache commands into skipped and still missing.

    Args:
        required: Every cache command the preflight knows about.
        missing: Commands for artifacts that are not cached yet.

    Returns:
        ``(skipped, needed)`` in ``required`` order.
    """
    missing_set = set(missing)
    skipped = [command for command in required if command not in missing_set]
    needed = [command for command in required if command in missing_set]
    return skipped, needed


def model_invocations(commands: list[str]) -> tuple[list[str], list[list[str]]]:
    """Split cache commands into ``ssv-models`` flags and other argv lists.

    Args:
        commands: Shell commands, each starting with ``ssv-models`` or another
            executable such as ``ollama``.

    Returns:
        Flags for one ``ssv-models`` process, and argv lists for the rest.
    """
    flags: list[str] = []
    other: list[list[str]] = []
    for command in commands:
        parts = shlex.split(command)
        if not parts:
            continue
        if parts[0] in _MODEL_BINARIES:
            flags.extend(parts[1:])
        else:
            other.append(parts)
    return flags, other


def prepare_models_argv(flags: list[str], executable: str) -> list[str]:
    """Return the ``ssv-models`` command that caches ``flags``.

    Args:
        flags: Arguments such as ``--clip`` and ``--dino``.
        executable: Python interpreter whose ``bin`` directory holds ``ssv-models``.

    Returns:
        Argv for the console script. ``prepare_models`` has no ``__main__``.
    """
    return [str(Path(executable).with_name("ssv-models")), *flags]


def _run(argv: list[str]) -> int:
    completed = subprocess.run(argv, check=False)
    return int(completed.returncode)


def main() -> None:
    """Cache missing local-pipeline models and skip the ones already cached."""
    args = local_pipeline_args()
    from ssv_vdp.local_env import apply_local_env

    apply_local_env(args)
    from selfsuvis.pipeline.core.logging import get_logger
    from selfsuvis.pipeline.core.preflight import run_local_preflight
    from selfsuvis.pipeline.vision.registry import auto_select, detect_resources

    log = get_logger("cache_models")
    resources = detect_resources()
    report = run_local_preflight(args, select_model=lambda task: auto_select(task, resources) or "")
    required = getattr(report, "cache_commands", None)
    if required is None:
        required = report.commands
    skipped, needed = pending_cache_commands(list(required), report.commands)
    for command in skipped:
        log.info("already cached, skipping: %s", command)
    if not needed:
        log.info("all required models are cached")
        return
    log.info("caching %d missing model command(s)", len(needed))
    for command in needed:
        log.info("cache: %s", command)
    flags, other = model_invocations(needed)
    status = 0
    if flags:
        status = _run(prepare_models_argv(flags, sys.executable))
    for argv in other:
        if status != 0:
            break
        status = _run(argv)
    if status != 0:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
