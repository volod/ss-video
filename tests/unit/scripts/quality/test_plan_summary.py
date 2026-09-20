from pathlib import Path

from selfsuvis.scripts.quality.plan_summary import main, summary_lines
from tests.unit.scripts.quality._plan_fixture import (
    human_block,
    later_block,
    plan_with,
    task_block,
    write_project,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_repository_plan_summary_names_the_next_task_per_lane() -> None:
    lines = summary_lines(PROJECT_ROOT)

    assert lines[0].startswith("tasks: ")
    assert any(line.startswith("next agent: ") for line in lines)
    assert any(line.startswith("next human: ") for line in lines)


def test_summary_counts_statuses_and_selects_required_work(tmp_path: Path) -> None:
    plan = plan_with(task_block("required-next"), task_block("optional-first", optional=True))
    root = write_project(tmp_path, plan=plan)

    lines = summary_lines(root)

    assert lines == [
        "tasks: 3",
        "agent lane: 3",
        "human lane: 0",
        "statuses: CLEAR=3",
        "eligible now: agent=3, human=0",
        "next agent: required-next [feature]",
        "next human: none",
    ]


def test_next_task_skips_work_with_open_start_dependencies(tmp_path: Path) -> None:
    plan = plan_with(
        task_block("blocked-first", dependencies="`build-later`."),
        later_blocks=(later_block(),),
    )
    root = write_project(tmp_path, plan=plan)

    assert "next agent: build-later [later]" in summary_lines(root)


def test_optional_dependencies_do_not_block(tmp_path: Path) -> None:
    plan = plan_with(task_block("soft-first", dependencies="none. Optional: `build-later`."))
    root = write_project(tmp_path, plan=plan)

    assert "next agent: soft-first [feature]" in summary_lines(root)


def test_lane_with_only_blocked_work_names_what_it_waits_on(tmp_path: Path) -> None:
    plan = plan_with(
        task_block(),
        human_blocks=(human_block(status="BLOCKED BY HUMAN", dependencies="`build-feature`."),),
    )
    root = write_project(tmp_path, plan=plan)

    lines = summary_lines(root)

    assert "eligible now: agent=2, human=0" in lines
    assert (
        "next human: none eligible; first in line `review-feature` waits on build-feature" in lines
    )


def test_invalid_plan_summary_refuses_to_schedule(tmp_path: Path) -> None:
    root = write_project(tmp_path, plan=plan_with())

    assert summary_lines(root) == ["plan is invalid; run make lint-spec-plan"]
    assert main(["--root", str(root)]) == 1


def test_main_succeeds_for_a_valid_plan(tmp_path: Path) -> None:
    assert main(["--root", str(write_project(tmp_path))]) == 0
