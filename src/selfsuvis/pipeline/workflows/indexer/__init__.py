"""VideoIndexer package — re-exports the public API of the original indexer module.

All external code that imports ``selfsuvis.pipeline.workflows.indexer.VideoIndexer``
continues to work unchanged.  Tests that monkeypatch ``idx_module.settings`` work
because ``settings`` is explicitly re-exported here.
"""

from selfsuvis.pipeline.core import (
    settings,  # noqa: F401 — required by tests via idx_module.settings
)

from ._core import VideoIndexer, _IndexFrameState  # noqa: F401

__all__ = ["VideoIndexer", "_IndexFrameState", "settings"]
