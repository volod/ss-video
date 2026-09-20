"""Validate relative links and Markdown heading anchors across project documentation.

Run: `python -m selfsuvis.scripts.quality.doc_links [--root DIR]` (or `make lint-doc-links`).
"""

from ss_kit.quality.doc_links import (
    anchors,
    broken_links,
    documentation_files,
    heading_anchor,
    main,
    relative_links,
)

__all__ = [
    "anchors",
    "broken_links",
    "documentation_files",
    "heading_anchor",
    "main",
    "relative_links",
]

if __name__ == "__main__":
    raise SystemExit(main())
