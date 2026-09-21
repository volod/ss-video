"""Paths for 4D analysis artifacts under ``$DATA_DIR``."""

from pathlib import Path

_SEGMENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")


def analysis_dir(mission_id: str) -> Path:
    """Return ``$DATA_DIR/analysis/<mission_id>/4d``.

    Args:
        mission_id: Single path segment. Slashes and parent references are rejected.

    Returns:
        The artifact directory. It is not created.
    """
    if not mission_id or mission_id in {".", ".."} or any(ch not in _SEGMENT for ch in mission_id):
        raise ValueError("mission_id must be a single path segment")
    from selfsuvis.pipeline.core import settings

    return settings.data_dir() / "analysis" / mission_id / "4d"


def benchmark_report_path() -> Path:
    """Return the default machine-readable benchmark report path."""
    from selfsuvis.pipeline.core import settings

    return settings.data_dir() / "analysis" / "_benchmark" / "report.json"
