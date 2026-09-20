"""Which unit tests GitHub CI collects.

Heavy tests import torch, OpenCV, ffmpeg, ONNX Runtime, or other vision/ML
stacks. `make test-unit` (full local venv) runs them. `make test-ci` passes
`--ci-light`, which skips collecting those paths so the GitHub job does not
install that footprint.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# POSIX paths relative to the repository root. A directory prefix ends with /.
HEAVY_UNIT_PATHS = (
    "tests/unit/pipeline/mapping/test_icp_fusion.py",
    "tests/unit/pipeline/media/test_rtsp_captioner.py",
    "tests/unit/pipeline/workflows/test_indexer_passes.py",
    "tests/unit/pipeline/workflows/test_model_version_payload.py",
    "tests/unit/app/routers/test_cvat_webhook_trigger.py",
)

BENCHMARK_UNIT_PATHS: tuple[str, ...] = ()

CI_LIGHT_MARKEXPR = (
    "not slow and not heavy and not benchmark and not gpu and not integration and not load"
)


def repo_relative(path: Path) -> str | None:
    """POSIX path relative to the repository root, or None if path is outside it."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def path_matches(relative: str, patterns: tuple[str, ...]) -> bool:
    """True if relative is one of the files or sits under a directory prefix."""
    for pattern in patterns:
        if pattern.endswith("/"):
            if relative == pattern[:-1] or relative.startswith(pattern):
                return True
        elif relative == pattern:
            return True
    return False


def is_heavy_unit_path(path: Path) -> bool:
    relative = repo_relative(path)
    return bool(relative) and path_matches(relative, HEAVY_UNIT_PATHS)


def is_benchmark_unit_path(path: Path) -> bool:
    relative = repo_relative(path)
    return bool(relative) and path_matches(relative, BENCHMARK_UNIT_PATHS)
