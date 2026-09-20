from pathlib import Path

from selfsuvis.scripts.quality.plan_model import (
    read_recorded_task_ids,
    read_registry,
    read_tasks,
)
from tests.unit.scripts.quality._plan_fixture import (
    SPEC,
    human_block,
    plan_with,
    task_block,
    write_project,
    write_record,
)


def test_registry_and_task_metadata_are_parsed(tmp_path: Path) -> None:
    root = write_project(tmp_path)

    registry = read_registry(root / "docs/design/spec.md")
    tasks = read_tasks(root / "docs/impl/plan.md")

    assert [capability.identifier for capability in registry] == ["foundation", "feature", "later"]
    assert registry[0].status == "shipped"
    assert [(task.identifier, task.serves, task.agent_status, task.group) for task in tasks] == [
        ("build-feature", "feature", "CLEAR", "feature"),
        ("build-later", "later", "CLEAR", "later"),
    ]


def test_continuation_lines_join_the_field_value(tmp_path: Path) -> None:
    block = task_block(dependencies="`build-later`; and\n  `other-task` on a wrapped line.")
    root = write_project(tmp_path, plan=plan_with(block))

    task = read_tasks(root / "docs/impl/plan.md")[0]

    assert task.fields["dependencies"] == ("`build-later`; and `other-task` on a wrapped line.")
    assert task.dependency_references == ["build-later", "other-task"]


def test_start_dependencies_stop_at_informational_markers(tmp_path: Path) -> None:
    dependencies = (
        "`hard-one`, `COOP_FLAG=false`. Optional: `soft-one`. Cross-lane note: `gate-one`."
    )
    root = write_project(tmp_path, plan=plan_with(task_block(dependencies=dependencies)))

    task = read_tasks(root / "docs/impl/plan.md")[0]

    assert task.start_dependencies == ["hard-one"]
    assert task.dependency_references == ["hard-one", "soft-one", "gate-one"]


def test_human_step_is_parsed_in_the_human_lane(tmp_path: Path) -> None:
    root = write_project(tmp_path, plan=plan_with(task_block(), human_blocks=(human_block(),)))

    human = [
        task
        for task in read_tasks(root / "docs/impl/plan.md")
        if task.agent_status == "HUMAN-GATED"
    ]

    assert human[0].section == "Human-Assisted Tasks"
    assert human[0].fields["human step"] == "An owner signs off on the result."


def test_fenced_task_examples_are_not_parsed(tmp_path: Path) -> None:
    plan = plan_with(task_block()) + "\n```markdown\n#### example-task\n```\n"
    root = write_project(tmp_path, plan=plan)

    identifiers = [task.identifier for task in read_tasks(root / "docs/impl/plan.md")]

    assert identifiers == ["build-feature", "build-later"]


def test_only_the_capability_registry_table_is_parsed(tmp_path: Path) -> None:
    spec = (
        SPEC
        + """
## Another table

| # | Capability | Status | How it is evaluated | Implementation |
| --- | --- | --- | --- | --- |
| 4 | `ignored` | planned | Other | - |
"""
    )
    root = write_project(tmp_path, spec=spec)

    identifiers = [item.identifier for item in read_registry(root / "docs/design/spec.md")]

    assert identifiers == ["foundation", "feature", "later"]


def test_recorded_task_ids_match_record_names(tmp_path: Path) -> None:
    write_record(tmp_path, "0001-split-governance-adopt-doc-lifecycle.md")
    write_record(tmp_path, "README.md")
    records = tmp_path / "docs/impl/records"

    found = read_recorded_task_ids(records, {"split-governance", "split"})

    # The `split` group yields `governance-adopt-doc-lifecycle`, never a bare suffix.
    assert found == {"adopt-doc-lifecycle", "governance-adopt-doc-lifecycle"}
    assert "doc-lifecycle" not in found
    assert read_recorded_task_ids(tmp_path / "absent", {"split-governance"}) == set()
