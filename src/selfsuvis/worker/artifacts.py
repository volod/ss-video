"""4D artifact directory used by worker handlers."""

from pathlib import Path

from selfsuvis.pipeline.analysis4d.paths import analysis_dir


def analysis_artifact_dir(mission_id: str) -> Path:
    """Return ``$DATA_DIR/analysis/<mission_id>/4d``.

    Handlers that write tracks, graph deltas, timelines, or QA records use this
    directory. They do not invent a second tree under maps or frames.
    """
    return analysis_dir(mission_id)
