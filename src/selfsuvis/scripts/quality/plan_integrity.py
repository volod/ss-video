"""Keep the product specification and forward plan as one coherent account.

Run: `python -m selfsuvis.scripts.quality.plan_integrity [--root DIR]` (or `make lint-spec-plan`).
"""

from ss_kit.quality.plan_integrity import (
    ADHOC_GROUP,
    AGENT_STATUSES,
    HUMAN_STATUSES,
    HUMAN_STEP,
    PLAN_DOC,
    PLANNED,
    RECORDS_DIR,
    REQUIRED_FIELDS,
    SHIPPED,
    SPEC_DOC,
    VALID_CAPABILITY_STATUSES,
    integrity_findings,
    main,
    recorded_task_ids,
)
from ss_kit.quality.plan_model import AGENT_SECTION, HUMAN_SECTION, PLAN_SECTIONS

__all__ = [
    "ADHOC_GROUP",
    "AGENT_SECTION",
    "AGENT_STATUSES",
    "HUMAN_SECTION",
    "HUMAN_STATUSES",
    "HUMAN_STEP",
    "PLAN_DOC",
    "PLAN_SECTIONS",
    "PLANNED",
    "RECORDS_DIR",
    "REQUIRED_FIELDS",
    "SHIPPED",
    "SPEC_DOC",
    "VALID_CAPABILITY_STATUSES",
    "integrity_findings",
    "main",
    "recorded_task_ids",
]

if __name__ == "__main__":
    raise SystemExit(main())
