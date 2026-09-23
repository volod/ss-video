"""Postflight strict verifier and Video-QA for one mission directory.

The job recomputes geometry, reviews only unsettled semantic claims, and
replaces ``timeline.json`` by naming the previous digest. Rejected and
uncertain claims stay in the audit log. They are not a fusion-rt payload.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from selfsuvis.pipeline.analysis4d.io import read_jsonl, read_model, sha256_bytes
from selfsuvis.pipeline.analysis4d.paths import analysis_dir
from selfsuvis.pipeline.analysis4d.qa import generate_qa, publishable_qa, qa_pairs
from selfsuvis.pipeline.analysis4d.review import ReviewProvider, load_review_provider
from selfsuvis.pipeline.analysis4d.schemas import (
    AnalysisManifest,
    ArtifactEntry,
    Degradation,
    GeometrySample,
    GraphDelta,
    Proposal,
    QaRecord,
    SceneTimeline,
    StageReport,
    TimelineEvent,
    TrackRecord,
)
from selfsuvis.pipeline.analysis4d.store import AnalysisStore
from selfsuvis.pipeline.analysis4d.verifier import (
    Resolution,
    VerifierOutput,
    audit_proposal,
    count_events,
    publishable_events,
    recheck_events,
    verify_proposals,
)
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)


@dataclass
class VerifyResult:
    """What one verifier run wrote or reused."""

    dest: Path
    resolutions: list[Resolution] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)
    qa: list[QaRecord] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    review_failure: str | None = None
    output: VerifierOutput | None = None


def run_mission_verify(
    mission_id: str,
    *,
    dest: Path | None = None,
    reviewer: ReviewProvider | None = None,
) -> VerifyResult:
    """Verify claims and write the timeline plus QA log.

    Args:
        mission_id: Mission whose 4D directory already has a manifest.
        dest: Artifact directory. The default is the mission 4D directory.
        reviewer: Semantic reviewer. The default follows
            ``ANALYSIS4D_REVIEW_PROVIDER`` and is unavailable.

    Returns:
        Resolutions, events, and QA. A second run with a verifier stage
        returns the existing timeline.

    Raises:
        FileNotFoundError: ``manifest.json`` or ``tracks.jsonl`` is missing.
    """
    target = Path(dest) if dest is not None else analysis_dir(mission_id)
    tracks_path = target / "tracks.jsonl"
    manifest_path = target / "manifest.json"
    if not tracks_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(target)
    before = tracks_path.read_bytes()
    manifest = read_model(manifest_path, AnalysisManifest)
    if any(stage.stage == "strict_verifier" for stage in manifest.stages):
        return _existing(target, manifest)
    tracks = read_jsonl(tracks_path, TrackRecord)
    deltas = read_jsonl(target / "graph-deltas.jsonl", GraphDelta)
    proposals = read_jsonl(target / "proposals.jsonl", Proposal)
    timeline = read_model(target / "timeline.json", SceneTimeline)
    samples = _load_geometry(target)
    frame = manifest.coordinate_frame
    metric = frame.metric_scale == "metric"
    chosen = reviewer if reviewer is not None else load_review_provider()
    started = time.perf_counter()
    output = verify_proposals(
        proposals,
        tracks=tracks,
        deltas=deltas,
        samples=samples,
        metric=metric,
        frame=frame.name,
        reviewer=chosen,
        events=list(timeline.events),
    )
    store = AnalysisStore(target)
    for proposal in proposals:
        resolution = next(
            item for item in output.resolutions if item.proposal_id == proposal.proposal_id
        )
        audit = audit_proposal(mission_id, proposal, resolution)
        if audit is not None:
            store.append_proposal(audit)
    events = recheck_events(
        list(timeline.events),
        deltas=deltas,
        samples=samples,
        tracks=tracks,
        metric=metric,
        root=target,
    )
    events.extend(
        count_events(
            output.resolutions,
            mission_id=mission_id,
            proposals=proposals,
            tracks=tracks,
            deltas=deltas,
            samples=samples,
            events=events,
            frame=frame.name,
            metric=metric,
            root=target,
        )
    )
    edges = [delta.edge for delta in deltas if delta.edge is not None]
    qa = generate_qa(mission_id, events, tracks, edges)
    for record in qa:
        store.append_qa(record)
    degradations = [
        Degradation(code="provider_unavailable", stage="strict_verifier", detail=_detail(output))
        for _flag in output.degradations
    ]
    _write_timeline(target, timeline, events, qa, degradations)
    elapsed = time.perf_counter() - started
    _write_manifest(target, manifest, degradations, elapsed, len(tracks))
    if tracks_path.read_bytes() != before:
        raise RuntimeError("strict verifier modified tracks.jsonl")
    logger.info(
        "strict verifier mission=%s events=%s qa=%s review_failure=%s",
        mission_id,
        len(publishable_events(events)),
        len(publishable_qa(qa)),
        output.review_failure or "",
    )
    return VerifyResult(
        dest=target,
        resolutions=output.resolutions,
        events=events,
        qa=qa,
        degradations=[item.code for item in degradations],
        review_failure=output.review_failure,
        output=output,
    )


def _existing(target: Path, manifest: AnalysisManifest) -> VerifyResult:
    timeline = read_model(target / "timeline.json", SceneTimeline)
    qa = read_jsonl(target / "qa.jsonl", QaRecord)
    flags = [item.code for item in manifest.degradations if item.stage == "strict_verifier"]
    return VerifyResult(
        dest=target,
        events=list(timeline.events),
        qa=qa,
        degradations=flags,
    )


def _write_timeline(
    dest: Path,
    timeline: SceneTimeline,
    events: list[TimelineEvent],
    qa: list[QaRecord],
    degradations: list[Degradation],
) -> None:
    path = dest / "timeline.json"
    previous = sha256_bytes(path.read_bytes())
    updated = timeline.model_copy(
        update={
            "events": events,
            "qa_pairs": qa_pairs(qa),
            "degradations": [*timeline.degradations, *degradations],
            "supersedes_sha256": previous,
        }
    )
    AnalysisStore(dest).write_timeline(updated)


def _write_manifest(
    dest: Path,
    manifest: AnalysisManifest,
    degradations: list[Degradation],
    elapsed: float,
    processed: int,
) -> None:
    path = dest / "manifest.json"
    previous = sha256_bytes(path.read_bytes())
    stage = StageReport(
        stage="strict_verifier",
        queue_delay_sec=0.0,
        inference_time_sec=elapsed,
        processed_frames=processed,
        skipped_frames=0,
        trigger_reason="postflight",
        degradation_flags=[item.code for item in degradations],
    )
    updated = manifest.model_copy(
        update={
            "artifacts": _refresh_artifacts(dest, list(manifest.artifacts)),
            "degradations": [*manifest.degradations, *degradations],
            "stages": [*manifest.stages, stage],
            "supersedes_sha256": previous,
        }
    )
    AnalysisStore(dest).write_manifest(updated)


def _refresh_artifacts(dest: Path, artifacts: list[ArtifactEntry]) -> list[ArtifactEntry]:
    refreshed: list[ArtifactEntry] = []
    for entry in artifacts:
        file_path = dest / entry.path
        if not file_path.is_file():
            refreshed.append(entry)
            continue
        payload = file_path.read_bytes()
        refreshed.append(
            entry.model_copy(update={"sha256": sha256_bytes(payload), "bytes": len(payload)})
        )
    return refreshed


def _load_geometry(dest: Path) -> list[GeometrySample]:
    directory = dest / "geometry"
    if not directory.is_dir():
        return []
    return [read_model(path, GeometrySample) for path in sorted(directory.rglob("*.json"))]


def _detail(output: VerifierOutput) -> str:
    if output.review_failure:
        return (
            f"review {output.review_failure}; deterministic results were kept "
            "and rejected claims were not published"
        )
    return "multimodal review unavailable; deterministic results were kept"
