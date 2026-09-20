"""High-level pipeline workflows and orchestration helpers.

Production exports (VideoIndexer, reporting) live here.
"""

from importlib import import_module

_EXPORTS = {
    "VideoIndexer": (".indexer", "VideoIndexer"),
    "generate_summary_html": (".reporting", "generate_summary_html"),
    "write_mission_report": (".reporting", "write_mission_report"),
    "latlon_bbox": ("selfsuvis.pipeline.analysis.change_detection", "latlon_bbox"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    package = __name__ if module_name.startswith(".") else None
    return getattr(import_module(module_name, package), attr_name)
