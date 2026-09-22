"""Strict-verifier and Video-QA benchmark.

``python -m selfsuvis.pipeline.analysis4d.verifier_benchmark`` scores the
pinned contradiction suite and the deep-profile scene graph. False acceptance
above 0.02, relation precision below 0.90, an accepted row without evidence,
or a reviewer failure that drops deterministic results fails the run.
"""

import argparse
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from selfsuvis.pipeline.analysis4d.graph import RegionBox, materialize
from selfsuvis.pipeline.analysis4d.graph_benchmark import (
    MISSION_ID as GRAPH_MISSION,
)
from selfsuvis.pipeline.analysis4d.graph_benchmark import (
    pinned_truth,
    prepare_pinned_scene,
)
from selfsuvis.pipeline.analysis4d.io import read_jsonl, read_model, write_bytes
from selfsuvis.pipeline.analysis4d.qa import execute_graph_program
from selfsuvis.pipeline.analysis4d.review import ReviewError, ScriptedReview, UnavailableReview
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_PROPOSAL,
    GeometrySample,
    GraphDelta,
    Proposal,
    SceneTimeline,
    TrackRecord,
)
from selfsuvis.pipeline.analysis4d.store import AnalysisStore
from selfsuvis.pipeline.analysis4d.validate import validate_bundle
from selfsuvis.pipeline.analysis4d.verifier import (
    accepted_edges,
    effective_status,
    publishable_events,
    verify_proposals,
)
from selfsuvis.pipeline.analysis4d.vlm import UnavailableVlm
from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.workflows.analysis4d_graph import run_mission_graph
from selfsuvis.pipeline.workflows.analysis4d_verify import run_mission_verify

logger = get_logger(__name__)

SCHEMA_VERIFIER_BENCHMARK = "ss-video.strict-verifier-benchmark.v1"
_FALSE_ACCEPT_MAX = 0.02
_RELATION_PRECISION_MIN = 0.90
_FALSE_PREDICATES = ("right_of", "above", "below", "contains", "intersects")
_SUITE_NEGATIVES = 50
_SUITE_POSITIVES = 10


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerifierBenchmarkReport(_Model):
    """Machine-readable result of the strict-verifier benchmark."""

    schema_version: str = SCHEMA_VERIFIER_BENCHMARK
    passed: bool
    false_accept_rate: float | None = None
    fixture_false_accept_rate: float | None = None
    relation_precision: float | None = None
    accepted_events_with_evidence: bool = False
    accepted_answers_with_evidence: bool = False
    fail_closed: bool = False
    qa_matches_program: bool = False
    uncertainty_gate: bool = False
    degradations: list[str] = Field(default_factory=list)
    qwen_probe: str = "not_requested"
    failures: list[str] = Field(default_factory=list)


def default_verifier_report_path() -> Path:
    """Return ``$DATA_DIR/analysis/_benchmark/verifier-report.json``."""
    from selfsuvis.pipeline.analysis4d.paths import benchmark_report_path

    return benchmark_report_path().with_name("verifier-report.json")


def run_verifier_benchmark(
    output: Path | None = None,
    *,
    probe_gpu: bool = False,
) -> VerifierBenchmarkReport:
    """Score contradiction claims, deep-profile relations, QA, and fail-closed review.

    Args:
        output: Directory for the deep-profile bundle. The default is beside the report.
        probe_gpu: When true, try the local Qwen-VL adapter. A missing model is
            recorded and does not fail the run.

    Returns:
        The report. ``passed`` is false when any gate fails.
    """
    root = (
        Path(output) if output is not None else default_verifier_report_path().parent / "verifier"
    )
    failures: list[str] = []
    fixture = _fixture_root()
    tracks = read_jsonl(fixture / "tracks.jsonl", TrackRecord)
    deltas = read_jsonl(fixture / "graph-deltas.jsonl", GraphDelta)
    samples = _load_geometry(fixture)
    proposals = read_jsonl(fixture / "proposals.jsonl", Proposal)
    manifest_frame = read_model(fixture / "timeline.json", SceneTimeline).coordinate_frame
    suite = _suite(proposals[0].mission_id)
    suite_out = verify_proposals(
        suite,
        tracks=tracks,
        deltas=deltas,
        samples=samples,
        metric=manifest_frame.metric_scale == "metric",
        frame=manifest_frame.name,
        reviewer=UnavailableReview(),
    )
    rate = _false_accept(
        suite_out.resolutions, [f"false-{index:03d}" for index in range(_SUITE_NEGATIVES)]
    )
    if rate > _FALSE_ACCEPT_MAX:
        failures.append(f"false acceptance {rate} exceeds {_FALSE_ACCEPT_MAX}")
    positives = [f"true-{index:03d}" for index in range(_SUITE_POSITIVES)]
    if any(_status(suite_out, item) != "accepted" for item in positives):
        failures.append("a true geometric claim was not accepted")
    fixture_out = verify_proposals(
        proposals,
        tracks=tracks,
        deltas=deltas,
        samples=samples,
        metric=True,
        frame=manifest_frame.name,
        reviewer=UnavailableReview(),
    )
    fixture_rate = _false_accept(fixture_out.resolutions, ["proposal-false"])
    if fixture_rate > _FALSE_ACCEPT_MAX:
        failures.append(f"fixture false acceptance {fixture_rate}")
    if _status(fixture_out, "proposal-true") != "accepted":
        failures.append("fixture count claim was not accepted from tracks")
    uncertain = _uncertain_gate(samples, tracks, deltas, manifest_frame.name)
    if not uncertain:
        failures.append("high residual did not keep a geometric claim uncertain")

    prepare_pinned_scene(root)
    run_mission_graph(
        GRAPH_MISSION,
        dest=root,
        provider=UnavailableVlm(),
        regions=[
            RegionBox("region-1", "loading-zone", [0.0, 0.0, 0.0], [10.0, 10.0, 4.0]),
        ],
    )
    verified = run_mission_verify(GRAPH_MISSION, dest=root, reviewer=UnavailableReview())
    try:
        bundle = validate_bundle(root)
    except Exception as exc:
        bundle = None
        failures.append(f"deep profile bundle invalid: {exc}")
    edges = accepted_edges(read_jsonl(root / "graph-deltas.jsonl", GraphDelta))
    precision = _precision(edges, pinned_truth()) if edges else 0.0
    if precision < _RELATION_PRECISION_MIN:
        failures.append(f"relation precision {precision} is below {_RELATION_PRECISION_MIN}")
    events_ok = _events_have_evidence(verified.events)
    answers_ok = _answers_have_evidence(verified.qa)
    if not events_ok:
        failures.append("an accepted event has no evidence")
    if not answers_ok:
        failures.append("an accepted answer has no evidence")
    programs_ok = _answers_match_programs(
        verified.events, verified.qa, read_jsonl(root / "tracks.jsonl", TrackRecord), edges
    )
    if not programs_ok:
        failures.append("a QA answer disagreed with its graph program")
    if not verified.qa:
        failures.append("deep profile produced no Video-QA from accepted state")
    if bundle is not None and bundle.manifest.profile != "deep":
        failures.append("deep profile bundle was not profile deep")

    closed = _fail_closed(fixture, root.parent / "verifier-fail-closed")
    if not closed:
        failures.append("timeout, refusal, or malformed review dropped deterministic results")
    probe = _probe_qwen() if probe_gpu else "not_requested"
    return VerifierBenchmarkReport(
        passed=not failures,
        false_accept_rate=rate,
        fixture_false_accept_rate=fixture_rate,
        relation_precision=precision,
        accepted_events_with_evidence=events_ok,
        accepted_answers_with_evidence=answers_ok,
        fail_closed=closed,
        qa_matches_program=programs_ok,
        uncertainty_gate=uncertain,
        degradations=list(verified.degradations),
        qwen_probe=probe,
        failures=failures,
    )


def _fixture_root() -> Path:
    from selfsuvis.pipeline.core import settings

    return Path(settings.PROJECT_ROOT) / "tests" / "assets" / "analysis4d" / "contradiction"


def _suite(mission_id: str) -> list[Proposal]:
    rows: list[Proposal] = []
    for index in range(_SUITE_NEGATIVES):
        predicate = _FALSE_PREDICATES[index % len(_FALSE_PREDICATES)]
        rows.append(
            _proposal(
                mission_id,
                f"false-{index:03d}",
                predicate,
                f"n-a is {predicate} n-b confidence 0.99",
                "accepted",
            )
        )
    for index in range(_SUITE_POSITIVES):
        rows.append(
            _proposal(
                mission_id,
                f"true-{index:03d}",
                "left_of",
                "n-a is left_of n-b",
                "uncertain",
            )
        )
    return rows


def _proposal(
    mission_id: str, proposal_id: str, predicate: str, text: str, status: str
) -> Proposal:
    return Proposal(
        schema_version=SCHEMA_PROPOSAL,
        mission_id=mission_id,
        proposal_id=proposal_id,
        claim_kind="relation",
        text=text,
        subject_id="n-a",
        predicate=predicate,
        object_id="n-b",
        start_sec=0.0,
        end_sec=1.0,
        verification_status=status,  # type: ignore[arg-type]
    )


def _uncertain_gate(samples, tracks, deltas, frame: str) -> bool:
    noisy = [sample.model_copy(update={"residual_m": 2.0}) for sample in samples]
    proposal = _proposal(
        samples[0].mission_id, "noisy-left", "left_of", "n-a is left_of n-b", "accepted"
    )
    output = verify_proposals(
        [proposal],
        tracks=tracks,
        deltas=deltas,
        samples=noisy,
        metric=True,
        frame=frame,
        reviewer=UnavailableReview(),
    )
    return (
        output.resolutions[0].status == "uncertain"
        and "uncertainty_gate" in output.resolutions[0].reasons
    )


def _fail_closed(fixture: Path, dest: Path) -> bool:
    ok = True
    for code in ("timeout", "refusal", "malformed"):
        mission = dest / code
        if mission.exists():
            shutil.rmtree(mission)
        shutil.copytree(fixture, mission)
        store = AnalysisStore(mission)
        store.append_proposal(
            Proposal(
                schema_version=SCHEMA_PROPOSAL,
                mission_id="mission-contradiction",
                proposal_id="proposal-semantic",
                claim_kind="attribute",
                text="weathered crate",
                verification_status="accepted",
            )
        )
        reviewer = ScriptedReview(error=ReviewError(code, code))
        result = run_mission_verify("mission-contradiction", dest=mission, reviewer=reviewer)
        proposals = read_jsonl(mission / "proposals.jsonl", Proposal)
        if result.review_failure != code:
            ok = False
        if effective_status(proposals, "proposal-semantic") != "uncertain":
            ok = False
        if effective_status(proposals, "proposal-false") == "accepted":
            ok = False
        edges = accepted_edges(read_jsonl(mission / "graph-deltas.jsonl", GraphDelta))
        if not any(edge.predicate == "left_of" for edge in edges):
            ok = False
        accepted = publishable_events(result.events)
        if any(event.verification.status != "accepted" or not event.evidence for event in accepted):
            ok = False
        try:
            validate_bundle(mission)
        except Exception:
            ok = False
        view = materialize(read_jsonl(mission / "graph-deltas.jsonl", GraphDelta))
        if not view.edges:
            ok = False
    return ok


def _false_accept(resolutions, negative_ids: list[str]) -> float:
    by_id = {item.proposal_id: item for item in resolutions}
    if not negative_ids:
        return 0.0
    accepted = sum(1 for item in negative_ids if by_id[item].status in {"accepted", "corrected"})
    return accepted / len(negative_ids)


def _status(output, proposal_id: str) -> str:
    return next(item.status for item in output.resolutions if item.proposal_id == proposal_id)


def _events_have_evidence(events) -> bool:
    accepted = [event for event in events if event.verification.status == "accepted"]
    return bool(accepted) and all(event.evidence for event in accepted)


def _answers_have_evidence(records) -> bool:
    accepted = [row for row in records if row.verification_status == "accepted"]
    return all(row.evidence_event_ids for row in accepted)


def _answers_match_programs(events, records, tracks, edges) -> bool:
    for record in records:
        if record.verification_status != "accepted":
            continue
        outcome = execute_graph_program(
            record.graph_program, events=events, tracks=tracks, edges=edges
        )
        if outcome is None or outcome.answer.value != record.answer.value:
            return False
        if not record.evidence_event_ids:
            return False
    return True


def _precision(edges, truth) -> float:
    if not edges:
        return 0.0
    used: set[int] = set()
    matched = 0
    for edge in edges:
        best = None
        best_iou = 0.0
        for index, row in enumerate(truth):
            if index in used:
                continue
            if (edge.subject_id, edge.predicate, edge.object_id) != (
                row.subject_id,
                row.predicate,
                row.object_id,
            ):
                continue
            inter = max(0.0, min(edge.end_sec, row.end_sec) - max(edge.start_sec, row.start_sec))
            union = max(edge.end_sec, row.end_sec) - min(edge.start_sec, row.start_sec)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best = index
        if best is not None and best_iou > 0:
            used.add(best)
            matched += 1
    return matched / len(edges)


def _load_geometry(root: Path) -> list[GeometrySample]:
    directory = root / "geometry"
    return [read_model(path, GeometrySample) for path in sorted(directory.rglob("*.json"))]


def _probe_qwen() -> str:
    from selfsuvis.pipeline.analysis4d.review import QwenVlReview, ReviewPacket

    provider = QwenVlReview()
    if provider.failed:
        logger.info("Qwen-VL unavailable: %s", provider.detail)
        return "unavailable"
    packet = ReviewPacket(
        proposal_id="probe-1",
        claim_kind="attribute",
        text="probe",
        measurements=["no measurement"],
        counter_evidence=["no counter-evidence"],
    )
    try:
        provider.review([packet])
    except ReviewError:
        return "fail_closed"
    return "parsed"


def main(argv: list[str] | None = None) -> int:
    """Write the strict-verifier benchmark report. Returns 0 when it passes."""
    parser = argparse.ArgumentParser(description="4D strict verifier and Video-QA benchmark")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--probe-gpu", action="store_true")
    args = parser.parse_args(argv)
    report_path = args.output or default_verifier_report_path()
    bundle_dir = args.bundle_dir or report_path.parent / "verifier"
    report = run_verifier_benchmark(bundle_dir, probe_gpu=args.probe_gpu)
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    write_bytes(report_path, payload.encode("utf-8"))
    logger.info("strict verifier benchmark passed=%s report=%s", report.passed, report_path)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
