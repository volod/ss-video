"""Summarize the forward plan and identify the next eligible work in each lane.

Run: `python -m selfsuvis.scripts.quality.plan_summary [--root DIR]` (or `make plan-status`).
"""

from ss_kit.quality.plan_summary import main, open_dependencies, summary_lines

__all__ = ["main", "open_dependencies", "summary_lines"]

if __name__ == "__main__":
    raise SystemExit(main())
