"""Fixture benchmark for the 4D contracts.

``python -m selfsuvis.pipeline.analysis4d.benchmark`` reads the pinned corpus,
validates every mission, and writes a JSON report. ``passed`` means the corpus
is contract-valid and every specification metric and stage field is present.
It does not apply the later quality thresholds.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from selfsuvis.pipeline.analysis4d.io import write_bytes
from selfsuvis.pipeline.analysis4d.metrics import score_case
from selfsuvis.pipeline.analysis4d.paths import benchmark_report_path
from selfsuvis.pipeline.analysis4d.schemas import (
    METRIC_NAMES,
    STAGE_FIELDS,
    BenchmarkReport,
    CaseResult,
    Degradation,
    MetricReport,
    StageReport,
)
from selfsuvis.pipeline.analysis4d.validate import ContractError, validate_bundle
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

_RUNNER_STAGE = "benchmark"


def default_corpus() -> Path:
    """Return ``tests/assets/analysis4d`` under the project root."""
    from selfsuvis.pipeline.core import settings

    return Path(settings.PROJECT_ROOT) / "tests" / "assets" / "analysis4d"


def process_gpu_memory_bytes() -> int | None:
    """Return this process's GPU memory in bytes, or 0 when it holds no context.

    Returns:
        None when ``nvidia-smi`` is unavailable. Zero is a measured result: the
        contract benchmark does not load a model.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    pid = str(os.getpid())
    used = 0
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == pid:
            used += int(float(parts[1])) * 1024 * 1024
    return used


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _aggregate(cases: list[CaseResult]) -> MetricReport:
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for case in cases:
        if not case.valid:
            continue
        for name in METRIC_NAMES:
            value = getattr(case.metrics, name)
            if value is not None:
                values[name].append(float(value))
    payload: dict[str, float | None] = {}
    for name, series in values.items():
        if not series:
            payload[name] = None
        elif name in {"identity_switches", "artifact_size_bytes", "queue_depth", "peak_vram_bytes"}:
            payload[name] = float(sum(series) if name != "queue_depth" else max(series))
        else:
            payload[name] = round(float(sum(series) / len(series)), 6)
    return MetricReport(**payload)


def run_benchmark(corpus: Path) -> BenchmarkReport:
    """Validate and score every mission listed in ``corpus.json``.

    Args:
        corpus: Directory containing ``corpus.json`` and one subdirectory per mission.

    Returns:
        A report whose metric object has every specification field.
    """
    corpus = Path(corpus)
    index = json.loads((corpus / "corpus.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    cases: list[CaseResult] = []
    degradations: list[Degradation] = []
    stages: list[StageReport] = []
    failures: list[str] = []
    seen_degradations: set[tuple[str, str, str]] = set()
    media_duration = 0.0
    for name in index["missions"]:
        mission_dir = corpus / name
        try:
            bundle = validate_bundle(mission_dir, require_truth=True)
        except ContractError as exc:
            failures.append(f"{name}: {exc}")
            cases.append(
                CaseResult(
                    mission_id=name,
                    valid=False,
                    empty_timeline=False,
                    conditions=[],
                    metrics=MetricReport(),
                    failures=exc.issues,
                )
            )
            continue
        assert bundle.truth is not None
        media_duration += bundle.truth.media_duration_sec
        metrics = score_case(bundle, artifact_bytes=_directory_bytes(mission_dir))
        empty = not bundle.timeline.events and not bundle.timeline.qa_pairs and not bundle.tracks
        cases.append(
            CaseResult(
                mission_id=bundle.manifest.mission_id,
                valid=True,
                empty_timeline=empty,
                conditions=list(bundle.truth.conditions),
                metrics=metrics,
            )
        )
        for item in bundle.manifest.degradations:
            key = (item.code, item.stage, item.detail)
            if key not in seen_degradations:
                seen_degradations.add(key)
                degradations.append(item)
        stages.extend(bundle.manifest.stages)
    elapsed = time.perf_counter() - started
    vram = process_gpu_memory_bytes()
    metrics = _aggregate(cases)
    metrics.stage_latency_sec = round(elapsed, 6)
    metrics.real_time_factor = round(elapsed / media_duration, 6) if media_duration else None
    metrics.peak_vram_bytes = None if vram is None else float(vram)
    covered: set[str] = set()
    for case in cases:
        if case.valid:
            covered.update(case.conditions)
    required = set(index.get("required_conditions", []))
    missing = sorted(required - covered)
    if missing:
        failures.append("missing conditions: " + ", ".join(missing))
    empty_ok = any(case.valid and case.empty_timeline for case in cases)
    if not empty_ok:
        failures.append("empty verified timeline was not accepted")
    runner = StageReport(
        stage=_RUNNER_STAGE,
        queue_delay_sec=0.0,
        inference_time_sec=round(elapsed, 6),
        processed_frames=len(cases),
        skipped_frames=sum(1 for case in cases if not case.valid),
        trigger_reason="fixture",
        degradation_flags=["models_missing"],
        queue_depth=0,
    )
    stages.append(runner)
    if vram is None:
        degradations.append(
            Degradation(
                code="models_missing",
                stage=_RUNNER_STAGE,
                detail="nvidia-smi did not report process GPU memory",
            )
        )
    report = BenchmarkReport(
        schema_version="ss-video.analysis4d-benchmark.v1",
        corpus_id=str(index["corpus_id"]),
        passed=not failures and empty_ok,
        empty_timeline_valid=empty_ok,
        metrics=metrics,
        degradations=degradations,
        stages=stages,
        cases=cases,
        failures=failures,
    )
    missing_metrics = [name for name in METRIC_NAMES if name not in MetricReport.model_fields]
    missing_stage = [name for name in STAGE_FIELDS if name not in StageReport.model_fields]
    if not report.stages:
        missing_stage.append("stages")
    if missing_metrics or missing_stage:
        report.passed = False
        report.failures.extend(missing_metrics + missing_stage)
    logger.info(
        "analysis4d benchmark corpus=%s passed=%s cases=%d latency_sec=%.4f",
        report.corpus_id,
        report.passed,
        len(report.cases),
        elapsed,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the fixture benchmark and print the JSON report."""
    parser = argparse.ArgumentParser(description="Score the pinned 4D analysis fixture corpus.")
    parser.add_argument("--corpus", type=Path, default=None, help="Corpus directory.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Report path. Default is under $DATA_DIR."
    )
    args = parser.parse_args(argv)
    corpus = args.corpus or default_corpus()
    report = run_benchmark(corpus)
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    output = args.output or benchmark_report_path()
    write_bytes(output, payload.encode("utf-8"))
    sys.stdout.write(payload)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
