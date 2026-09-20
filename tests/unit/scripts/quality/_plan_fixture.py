"""Synthetic specification and plan documents for the quality-check tests."""

from pathlib import Path

SPEC = """# Design

## Capability Registry

| # | Capability | Status | How it is evaluated | Implementation |
| --- | --- | --- | --- | --- |
| 1 | `foundation` | shipped | Fresh setup passes | [Foundation](../impl/current.md) |
| 2 | `feature` | planned | Fixture behavior passes | -- |
| 3 | `later` | planned | Later behavior passes | -- |
"""


def task_block(
    identifier: str = "build-feature",
    *,
    capability: str = "feature",
    status: str = "CLEAR",
    optional: bool = False,
    dependencies: str = "none.",
    human_step: str | None = None,
) -> str:
    suffix = " (optional)" if optional else ""
    human = f"- Human step: {human_step}\n" if human_step is not None else ""
    return f"""#### {identifier}{suffix}

Describe future product behavior.

- Serves: `{capability}` -- [Feature](../design/spec.md#feature)
- Agent status: {status}
- Dependencies: {dependencies}
- User-visible outcome: A user can exercise the feature.
{human}- Scope boundary: The fixture path only; deployment is outside scope.
- Data and artifact paths: `tests/fixtures/` only.
- Execution path: Add a module and deterministic tests.
- Acceptance gates: `make lint` passes or the negative result is recorded.
- Documentation target: [Current state](current.md).
"""


def human_block(
    identifier: str = "review-feature",
    *,
    status: str = "HUMAN-GATED",
    dependencies: str = "none.",
    human_step: str | None = "An owner signs off on the result.",
) -> str:
    return task_block(identifier, status=status, dependencies=dependencies, human_step=human_step)


def later_block(identifier: str = "build-later", *, dependencies: str = "none.") -> str:
    return task_block(identifier, capability="later", dependencies=dependencies)


def plan_with(
    *blocks: str,
    human_blocks: tuple[str, ...] = (),
    later_blocks: tuple[str, ...] = (later_block(),),
    later_first: bool = False,
) -> str:
    agent_content = "\n".join(blocks) if blocks else "No open agent tasks."
    human_content = "\n".join(human_blocks) if human_blocks else "No open human tasks."
    feature = f"### Feature -- `feature`\n\n{agent_content}\n"
    later = f"### Later -- `later`\n\n{chr(10).join(later_blocks)}\n" if later_blocks else ""
    groups = f"{later}\n{feature}" if later_first else f"{feature}\n{later}"
    return f"""# Plan

## Agent Implementation Tasks

{groups}
## Human-Assisted Tasks

### Feature -- `feature`

{human_content}
"""


def write_project(root: Path, *, spec: str = SPEC, plan: str | None = None) -> Path:
    (root / "docs/design").mkdir(parents=True)
    (root / "docs/impl").mkdir(parents=True)
    (root / "docs/design/spec.md").write_text(spec, encoding="utf-8")
    resolved_plan = plan if plan is not None else plan_with(task_block())
    (root / "docs/impl/plan.md").write_text(resolved_plan, encoding="utf-8")
    return root


def write_record(root: Path, name: str) -> None:
    records = root / "docs/impl/records"
    records.mkdir(parents=True, exist_ok=True)
    (records / name).write_text("# Record\n", encoding="utf-8")
