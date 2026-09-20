from pathlib import Path

import pytest

from selfsuvis.scripts.quality.project_root import discover_project_root


def test_project_root_is_discovered_from_a_nested_directory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("", encoding="utf-8")
    nested = tmp_path / "src/package"
    nested.mkdir(parents=True)

    assert discover_project_root(nested) == tmp_path


def test_project_root_reports_missing_markers(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no project root"):
        discover_project_root(tmp_path)
