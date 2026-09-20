from pathlib import Path

import pytest

from selfsuvis.scripts.quality.plan_integrity import integrity_findings, main
from tests.unit.scripts.quality._plan_fixture import (
    SPEC,
    human_block,
    later_block,
    plan_with,
    task_block,
    write_project,
    write_record,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _findings(tmp_path: Path, plan: str, spec: str = SPEC) -> list[str]:
    return integrity_findings(write_project(tmp_path, spec=spec, plan=plan))


def _has(findings: list[str], expected: str) -> bool:
    return any(expected in finding for finding in findings)


def test_repository_specification_and_plan_agree() -> None:
    assert integrity_findings(PROJECT_ROOT) == []


def test_a_healthy_plan_has_no_findings(tmp_path: Path) -> None:
    plan = plan_with(task_block(), human_blocks=(human_block(),))

    assert _findings(tmp_path, plan) == []


def test_unknown_capability_is_reported(tmp_path: Path) -> None:
    plan = plan_with(task_block()).replace("- Serves: `feature`", "- Serves: `unknown`")

    assert _has(_findings(tmp_path, plan), "serves unregistered capability `unknown`")


def test_misfiled_task_is_reported(tmp_path: Path) -> None:
    plan = plan_with(task_block(), task_block("stray-task", capability="later"))

    assert _has(_findings(tmp_path, plan), "`stray-task`: serves `later` under `feature`")


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("- Agent status: CLEAR", "- Agent status:", "Agent Status"),
        ("- Acceptance gates:", "- Other gates:", "Acceptance Gates"),
        ("- Documentation target:", "- Docs:", "Documentation Target"),
    ],
)
def test_missing_field_is_named(tmp_path: Path, old: str, new: str, expected: str) -> None:
    plan = plan_with(task_block()).replace(old, new)

    assert _has(_findings(tmp_path, plan), f"missing non-empty `{expected}` field")


def test_human_status_in_the_agent_lane_is_rejected(tmp_path: Path) -> None:
    plan = plan_with(task_block(status="HUMAN-GATED"))

    assert _has(_findings(tmp_path, plan), "invalid in 'Agent Implementation Tasks'")


def test_agent_status_in_the_human_lane_is_rejected(tmp_path: Path) -> None:
    plan = plan_with(task_block(), human_blocks=(human_block(status="CLEAR"),))

    assert _has(_findings(tmp_path, plan), "invalid in 'Human-Assisted Tasks'")


def test_human_lane_task_requires_a_human_step(tmp_path: Path) -> None:
    plan = plan_with(task_block(), human_blocks=(human_block(human_step=None),))

    assert _has(_findings(tmp_path, plan), "missing a non-empty `Human Step` field")


def test_agent_lane_task_must_not_declare_a_human_step(tmp_path: Path) -> None:
    plan = plan_with(task_block(human_step="Someone clicks a button."))

    assert _has(_findings(tmp_path, plan), "agent-lane task declares a `Human Step`")


def test_out_of_order_group_is_reported(tmp_path: Path) -> None:
    plan = plan_with(task_block(), later_first=True)

    assert _has(_findings(tmp_path, plan), "do not follow registry order")


def test_required_work_cannot_follow_optional_work(tmp_path: Path) -> None:
    plan = plan_with(task_block("nice-to-have", optional=True), task_block("required-work"))

    assert _has(_findings(tmp_path, plan), "`required-work` follows optional work")


def test_planned_capability_requires_a_task(tmp_path: Path) -> None:
    assert _has(_findings(tmp_path, plan_with()), "`feature`: planned but no forward task")


def test_shipped_capability_requires_a_current_link(tmp_path: Path) -> None:
    spec = SPEC.replace("[Foundation](../impl/current.md)", "none")

    assert _has(_findings(tmp_path, plan_with(task_block()), spec), "no current-documentation")


def test_capability_requires_an_evaluation(tmp_path: Path) -> None:
    spec = SPEC.replace("| Later behavior passes |", "| -- |")

    assert _has(_findings(tmp_path, plan_with(task_block()), spec), "no evaluation declares")


def test_forward_plan_rejects_history_and_dates(tmp_path: Path) -> None:
    plan = plan_with(task_block()).replace(
        "Describe future product behavior.", "DONE on 2026-01-01."
    )

    assert _has(_findings(tmp_path, plan), "history or a date")


def test_malformed_task_heading_is_reported(tmp_path: Path) -> None:
    plan = plan_with(task_block()).replace("#### build-feature", "#### Build feature")

    assert _has(_findings(tmp_path, plan), "malformed task heading")


def test_duplicate_task_id_is_reported(tmp_path: Path) -> None:
    plan = plan_with(task_block(), task_block())

    assert _has(_findings(tmp_path, plan), "task id is duplicated")


def test_unknown_dependency_is_reported(tmp_path: Path) -> None:
    plan = plan_with(task_block(dependencies="`missing-task`."))

    assert _has(_findings(tmp_path, plan), "depends on `missing-task`, which is neither")


def test_dependency_on_a_recorded_finished_task_is_accepted(tmp_path: Path) -> None:
    root = write_project(tmp_path, plan=plan_with(task_block(dependencies="`old-task`.")))
    write_record(root, "0001-feature-old-task.md")

    assert integrity_findings(root) == []


def test_multi_line_dependency_is_checked(tmp_path: Path) -> None:
    plan = plan_with(task_block(dependencies="`build-later`, and on a wrapped\n  `missing-task`."))

    assert _has(_findings(tmp_path, plan), "depends on `missing-task`")


def test_start_dependency_cycle_is_reported(tmp_path: Path) -> None:
    plan = plan_with(
        task_block(dependencies="`build-later`."),
        later_blocks=(later_block(dependencies="`build-feature`."),),
    )

    assert _has(_findings(tmp_path, plan), "start-dependency cycle")


def test_cross_lane_note_does_not_form_a_cycle(tmp_path: Path) -> None:
    plan = plan_with(
        task_block(dependencies="none. Cross-lane note: evidence comes from `review-feature`."),
        human_blocks=(human_block(dependencies="`build-feature`."),),
    )

    assert _findings(tmp_path, plan) == []


def test_missing_documents_are_reported(tmp_path: Path) -> None:
    assert integrity_findings(tmp_path) == [
        "missing required document: docs/design/spec.md",
        "missing required document: docs/impl/plan.md",
    ]


def test_main_returns_failure_for_invalid_documents(tmp_path: Path) -> None:
    write_project(tmp_path, plan=plan_with())

    assert main(["--root", str(tmp_path)]) == 1


def test_main_returns_success_for_valid_documents(tmp_path: Path) -> None:
    write_project(tmp_path)

    assert main(["--root", str(tmp_path)]) == 0
