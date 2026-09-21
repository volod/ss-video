"""Pinned corpus benchmark report."""

import json
from pathlib import Path

from selfsuvis.pipeline.analysis4d.benchmark import main, run_benchmark
from selfsuvis.pipeline.analysis4d.schemas import DEGRADATION_FIELDS, METRIC_NAMES, STAGE_FIELDS

ASSETS = Path(__file__).resolve().parents[4] / "tests" / "assets" / "analysis4d"


def test_benchmark_reports_every_metric_and_accepts_empty_timeline(tmp_path: Path) -> None:
    report = run_benchmark(ASSETS)
    payload = report.model_dump(mode="json")
    assert report.passed
    assert report.empty_timeline_valid
    assert set(METRIC_NAMES) <= set(payload["metrics"])
    assert payload["degradations"]
    for item in payload["degradations"]:
        assert set(DEGRADATION_FIELDS) <= set(item)
    assert payload["stages"]
    for stage in payload["stages"]:
        assert set(STAGE_FIELDS) <= set(stage)
    empty = next(case for case in report.cases if case.mission_id == "mission-empty")
    assert empty.valid and empty.empty_timeline
    nominal = next(case for case in report.cases if case.mission_id == "mission-nominal")
    assert nominal.metrics.relation_precision == 1
    assert nominal.metrics.qa_accuracy == 1
    assert nominal.metrics.event_f1 == 1
    occlusion = next(case for case in report.cases if case.mission_id == "mission-occlusion")
    assert occlusion.metrics.identity_switches == 1
    contradiction = next(
        case for case in report.cases if case.mission_id == "mission-contradiction"
    )
    assert contradiction.metrics.verifier_false_accept_rate == 0
    assert contradiction.metrics.verifier_false_reject_rate == 0
    assert main(["--corpus", str(ASSETS), "--output", str(tmp_path / "report.json")]) == 0
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written["passed"] is True
    assert set(METRIC_NAMES) <= set(written["metrics"])
