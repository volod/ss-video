"""GitHub CI collects the light unit suite, not torch/cv2/ffmpeg tests."""

from pathlib import Path

import pytest

from tests.support.ci_light import BENCHMARK_UNIT_PATHS, HEAVY_UNIT_PATHS, REPO_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_heavy_unit_paths_exist() -> None:
    missing = [path for path in HEAVY_UNIT_PATHS if not (REPO_ROOT / path.rstrip("/")).exists()]
    assert missing == []


def test_benchmark_unit_paths_exist() -> None:
    missing = [path for path in BENCHMARK_UNIT_PATHS if not (REPO_ROOT / path.rstrip("/")).exists()]
    assert missing == []


def test_root_ci_unit_job_is_the_light_suite() -> None:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((PROJECT_ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["unit"]["steps"])
    assert "make test-ci" in commands
    assert "make test-unit" not in commands
    assert "ffmpeg" not in commands
    assert "torch" not in commands
    assert "opencv" not in commands


def test_root_ci_has_no_staged_project_jobs() -> None:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((PROJECT_ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
    assert "projects" not in workflow["jobs"]
    assert "projects-matrix" not in workflow["jobs"]


def test_fusion_rt_sidecar_image_is_slim() -> None:
    compose = (PROJECT_ROOT / "docker/core/docker-compose.yml").read_text("utf-8")
    assert "Dockerfile.fusion_rt" in compose
    tests_df = (PROJECT_ROOT / "docker/test/Dockerfile.tests").read_text("utf-8")
    fusion_df = (PROJECT_ROOT / "docker/core/Dockerfile.fusion_rt").read_text("utf-8")
    assert "install-python.sh --runtime" in fusion_df
    assert ".[vision]" not in fusion_df
    assert ".[vision]" not in tests_df
    assert ".[dev]" not in tests_df
    assert "tests/test_fusion_rt.py" in tests_df
