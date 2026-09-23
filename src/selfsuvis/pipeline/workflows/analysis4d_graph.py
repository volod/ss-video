"""Postflight temporal scene graph for one mission directory.

The job reads ``tracks.jsonl`` and ``geometry/`` and appends graph deltas.
An unavailable VLM records ``provider_unavailable`` and still writes the
deterministic graph. Track bytes are not rewritten.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from selfsuvis.pipeline.analysis4d.graph import (
    GraphView,
    RegionBox,
    materialize,
    reduce_graph,
    serialize_context,
)
from selfsuvis.pipeline.analysis4d.io import (
    canonical_bytes,
    read_jsonl,
    read_model,
    sha256_bytes,
    write_bytes,
)
from selfsuvis.pipeline.analysis4d.paths import analysis_dir
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_GEOMETRY,
    AnalysisManifest,
    ArtifactEntry,
    Degradation,
    GeometrySample,
    GraphDelta,
    Proposal,
    SceneTimeline,
    StageReport,
    TimelineEvent,
    TrackRecord,
)
from selfsuvis.pipeline.analysis4d.store import AnalysisStore
from selfsuvis.pipeline.analysis4d.vlm import (
    ProposalProvider,
    load_proposal_provider,
    proposals_from_claims,
)
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

_QUAT = [0.0, 0.0, 0.0, 1.0]


@dataclass
class GraphResult:
    """What one scene-graph run wrote or reused."""

    dest: Path
    deltas: list[GraphDelta] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    view: GraphView = field(default_factory=GraphView)
    tracks_sha256: str = ""


def run_mission_graph(
    mission_id: str,
    *,
    dest: Path | None = None,
    provider: ProposalProvider | None = None,
    regions: list[RegionBox] | None = None,
) -> GraphResult:
    """Reduce tracks and geometry into an append-only scene graph.

    Args:
        mission_id: Mission whose 4D directory already has tracks and a manifest.
        dest: Artifact directory. The default is the mission 4D directory.
        provider: Proposal provider. The default follows ``ANALYSIS4D_VLM_PROVIDER``.
        regions: Static regions written when that subject has no geometry yet.

    Returns:
        Deltas, proposals, events, and the materialized view.

    Raises:
        FileNotFoundError: ``tracks.jsonl`` or ``manifest.json`` is missing.
    """
    target = Path(dest) if dest is not None else analysis_dir(mission_id)
    tracks_path = target / "tracks.jsonl"
    manifest_path = target / "manifest.json"
    if not tracks_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(target)
    before = tracks_path.read_bytes()
    manifest = read_model(manifest_path, AnalysisManifest)
    if any(stage.stage == "scene_graph" for stage in manifest.stages):
        return _existing(target, before)
    tracks = read_jsonl(tracks_path, TrackRecord)
    samples = _load_geometry(target)
    frame = manifest.coordinate_frame
    metric = frame.metric_scale == "metric"
    region_ids = {region.node_id for region in regions or []}
    samples.extend(
        _write_regions(
            target,
            mission_id,
            regions or [],
            stamps=_stamps(samples, tracks),
            frame_name=frame.name,
            metric=metric,
            calibration_id=frame.calibration_id,
        )
    )
    started = time.perf_counter()
    reduced = reduce_graph(
        mission_id,
        tracks,
        samples,
        frame_name=frame.name,
        metric=metric,
        region_ids=region_ids,
    )
    elapsed = time.perf_counter() - started
    view = materialize(reduced.deltas)
    context = serialize_context(tracks, samples, view)
    chosen = provider if provider is not None else load_proposal_provider()
    claims, degraded = _propose(chosen, context)
    proposals = proposals_from_claims(mission_id, claims, samples)
    flags = _flags(tracks, samples, degraded)
    degradations = [
        Degradation(code=code, stage="scene_graph", detail=_detail(code)) for code in flags
    ]
    store = AnalysisStore(target)
    for delta in reduced.deltas:
        store.append_delta(delta)
    for proposal in proposals:
        store.append_proposal(proposal)
    _write_timeline(target, manifest, reduced.events, degradations)
    _write_manifest(target, manifest, degradations, elapsed, len(_stamps(samples, tracks)))
    after = tracks_path.read_bytes()
    if after != before:
        raise RuntimeError("scene graph modified tracks.jsonl")
    logger.info(
        "scene graph mission=%s deltas=%s proposals=%s events=%s",
        mission_id,
        len(reduced.deltas),
        len(proposals),
        len(reduced.events),
    )
    return GraphResult(
        dest=target,
        deltas=reduced.deltas,
        proposals=proposals,
        events=reduced.events,
        degradations=flags,
        view=view,
        tracks_sha256=sha256_bytes(before),
    )


def _propose(provider: ProposalProvider, context: dict) -> tuple[list, bool]:
    if getattr(provider, "failed", False):
        return [], True
    try:
        return list(provider.propose(context)), False
    except Exception:
        logger.warning("VLM proposal provider failed; deterministic graph kept")
        return [], True


def _existing(target: Path, track_bytes: bytes) -> GraphResult:
    deltas = read_jsonl(target / "graph-deltas.jsonl", GraphDelta)
    proposals = read_jsonl(target / "proposals.jsonl", Proposal)
    timeline = read_model(target / "timeline.json", SceneTimeline)
    return GraphResult(
        dest=target,
        deltas=deltas,
        proposals=proposals,
        events=list(timeline.events),
        degradations=[item.code for item in timeline.degradations],
        view=materialize(deltas),
        tracks_sha256=sha256_bytes(track_bytes),
    )


def _write_regions(
    dest: Path,
    mission_id: str,
    regions: list[RegionBox],
    *,
    stamps: list[float],
    frame_name: str,
    metric: bool,
    calibration_id: str | None,
) -> list[GeometrySample]:
    written: list[GeometrySample] = []
    scale = "metric" if metric else "unavailable"
    for region in regions:
        for stamp in stamps:
            millis = int(round(stamp * 1000))
            path = dest / "geometry" / region.node_id / f"{millis:07d}.json"
            sample = GeometrySample(
                schema_version=SCHEMA_GEOMETRY,
                mission_id=mission_id,
                sample_id=f"geo-{region.node_id}-{millis:07d}",
                subject_id=region.node_id,
                t_sec=stamp,
                frame=frame_name,
                center_m=list(region.center_m),
                extent_m=list(region.extent_m),
                quaternion_xyzw=list(_QUAT),
                metric_scale=scale,  # type: ignore[arg-type]
                calibration_id=calibration_id if metric else None,
                observation_count=1,
            )
            if not path.is_file():
                write_bytes(path, canonical_bytes(sample))
            written.append(sample)
    return written


def _write_timeline(
    dest: Path,
    manifest: AnalysisManifest,
    events: list[TimelineEvent],
    degradations: list[Degradation],
) -> None:
    path = dest / "timeline.json"
    timeline = read_model(path, SceneTimeline)
    previous = sha256_bytes(path.read_bytes())
    updated = timeline.model_copy(
        update={
            "events": events,
            "degradations": [*timeline.degradations, *degradations],
            "supersedes_sha256": previous,
        }
    )
    AnalysisStore(dest).write_timeline(updated)
    del manifest


def _write_manifest(
    dest: Path,
    manifest: AnalysisManifest,
    degradations: list[Degradation],
    elapsed: float,
    processed: int,
) -> None:
    path = dest / "manifest.json"
    previous = sha256_bytes(path.read_bytes())
    artifacts = _refresh_artifacts(dest, list(manifest.artifacts))
    stage = StageReport(
        stage="scene_graph",
        queue_delay_sec=0.0,
        inference_time_sec=elapsed,
        processed_frames=processed,
        skipped_frames=0,
        trigger_reason="postflight",
        degradation_flags=[item.code for item in degradations],
    )
    updated = manifest.model_copy(
        update={
            "artifacts": artifacts,
            "degradations": [*manifest.degradations, *degradations],
            "stages": [*manifest.stages, stage],
            "supersedes_sha256": previous,
        }
    )
    AnalysisStore(dest).write_manifest(updated)


def _refresh_artifacts(dest: Path, artifacts: list[ArtifactEntry]) -> list[ArtifactEntry]:
    refreshed: list[ArtifactEntry] = []
    known = set()
    for entry in artifacts:
        file_path = dest / entry.path
        if not file_path.is_file():
            refreshed.append(entry)
            known.add(entry.path)
            continue
        payload = file_path.read_bytes()
        refreshed.append(
            entry.model_copy(update={"sha256": sha256_bytes(payload), "bytes": len(payload)})
        )
        known.add(entry.path)
    for file_path in sorted((dest / "geometry").rglob("*.json")):
        rel = file_path.relative_to(dest).as_posix()
        if rel in known:
            continue
        payload = file_path.read_bytes()
        refreshed.append(
            ArtifactEntry(
                path=rel,
                kind="geometry",
                schema_version=SCHEMA_GEOMETRY,
                sha256=sha256_bytes(payload),
                bytes=len(payload),
            )
        )
    return refreshed


def _load_geometry(dest: Path) -> list[GeometrySample]:
    directory = dest / "geometry"
    if not directory.is_dir():
        return []
    return [read_model(path, GeometrySample) for path in sorted(directory.rglob("*.json"))]


def _stamps(samples: list[GeometrySample], tracks: list[TrackRecord]) -> list[float]:
    found = {round(sample.t_sec, 6) for sample in samples}
    found.update(round(record.t_sec, 6) for record in tracks)
    return sorted(found)


def _flags(tracks: list[TrackRecord], samples: list[GeometrySample], degraded: bool) -> list[str]:
    flags: list[str] = []
    if degraded:
        flags.append("provider_unavailable")
    if not tracks and not samples:
        flags.append("empty_scene")
    return flags


def _detail(code: str) -> str:
    return {
        "provider_unavailable": "compact VLM unavailable; deterministic graph was kept",
        "empty_scene": "no tracks or geometry were available for the scene graph",
    }.get(code, code)
