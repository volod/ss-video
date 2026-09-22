"""Causal fast profile and postflight deep profile for one mission.

The fast pass keeps track continuity when the queue is under pressure and
publishes only accepted events. The deep pass appends a revision and points
the manifest and timeline at the fast digest. A second call with the same
frames does not publish those events again.
"""

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from selfsuvis.pipeline.analysis4d.budget import (
    BoundedFrameQueue,
    BudgetController,
    DiscardedSpan,
)
from selfsuvis.pipeline.analysis4d.graph import RegionBox, reduce_graph
from selfsuvis.pipeline.analysis4d.io import (
    canonical_bytes,
    read_jsonl,
    read_model,
    sha256_bytes,
    write_bytes,
)
from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal
from selfsuvis.pipeline.analysis4d.passes import _gap_records, run_pass, write_profiles
from selfsuvis.pipeline.analysis4d.paths import analysis_dir
from selfsuvis.pipeline.analysis4d.pin import active_grounding, active_mask, selector_config
from selfsuvis.pipeline.analysis4d.profile import (
    chunk_seconds,
    fast_event_types,
    gpu_slots,
    queue_capacity,
    sensor_id,
    zone_id,
)
from selfsuvis.pipeline.analysis4d.providers import Prompt
from selfsuvis.pipeline.analysis4d.publish import publish_accepted, published_ids
from selfsuvis.pipeline.analysis4d.review import UnavailableReview
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_GAP,
    SCHEMA_GEOMETRY,
    SCHEMA_TRACK,
    AnalysisManifest,
    ArtifactEntry,
    Box2D,
    CoordinateFrame,
    Degradation,
    GapRecord,
    GeometrySample,
    GraphDelta,
    IdentityLink,
    SceneTimeline,
    StageReport,
    TimelineEvent,
    TrackRecord,
)
from selfsuvis.pipeline.analysis4d.store import AnalysisStore
from selfsuvis.pipeline.analysis4d.tracks import Observation, TrackerConfig
from selfsuvis.pipeline.analysis4d.validate import validate_bundle
from selfsuvis.pipeline.analysis4d.vlm import UnavailableVlm
from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.workflows.analysis4d_graph import run_mission_graph
from selfsuvis.pipeline.workflows.analysis4d_tracks import _load_grounding, _load_mask
from selfsuvis.pipeline.workflows.analysis4d_verify import run_mission_verify

logger = get_logger(__name__)

STATE_NAME = "orchestration-state.json"
_QUAT = [0.0, 0.0, 0.0, 1.0]
_REGION = RegionBox(
    node_id="region-loading",
    label="loading-zone",
    center_m=[0.0, 0.0, 0.0],
    extent_m=[4.0, 4.0, 4.0],
)


@dataclass
class ProfileRun:
    """One fast or deep orchestration result."""

    dest: Path
    profile: str
    published_event_ids: list[str] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    backlog: list[str] = field(default_factory=list)
    queue_depth: int = 0
    queue_capacity: int = 0
    real_time_factor: float = 0.0
    lags_sec: list[float] = field(default_factory=list)
    gap_ids: list[str] = field(default_factory=list)
    replayed: bool = False
    superseded: bool = False
    media_duration_sec: float = 0.0


def run_fast_profile(
    mission_id: str,
    frames: list[FrameSignal],
    prompts: list[Prompt],
    *,
    dest: Path | None = None,
    grounding: object | None = None,
    prepare_geometry: Callable[[Path], None] | None = None,
    fixture_geometry: bool = False,
    regions: list[RegionBox] | None = None,
    duration_sec: float | None = None,
    capacity: int | None = None,
    slots: int | None = None,
    published_at: datetime | None = None,
) -> ProfileRun:
    """Run the causal fast DAG and publish accepted events.

    Args:
        mission_id: Mission id. Also the default artifact directory name.
        frames: Decoded frames in non-decreasing timestamp order.
        prompts: Grounding prompts.
        dest: Artifact directory. The default is the mission 4D directory.
        grounding: Keyframe detector. The default follows the pinned provider.
        prepare_geometry: Optional dense-geometry callback. Skipped under pressure.
        fixture_geometry: Write one box per track so the graph can emit events.
        regions: Regions passed to the scene graph when fixture geometry is on.
        duration_sec: Media length used for the real-time factor. The default
            is the span of ``frames``.
        capacity: Queue capacity. The default is ``ANALYSIS4D_QUEUE_CAPACITY``.
        slots: Optional GPU slots. The default is ``ANALYSIS4D_GPU_SLOTS``.
        published_at: Envelope timestamp. The default is the current UTC time.

    Returns:
        Queue, lag, and publication details. A second call with the same frames
        returns the existing publication and writes nothing new.
    """
    target = Path(dest) if dest is not None else analysis_dir(mission_id)
    target.mkdir(parents=True, exist_ok=True)
    digest = _input_digest(frames, prompts)
    state = _read_state(target)
    if state.get("fast_done") and state.get("input_digest") == digest:
        return _from_state(target, state, profile="fast", replayed=True)
    if state.get("input_digest") not in (None, digest):
        raise ValueError("mission already has a 4D run for a different input")

    started = time.perf_counter()
    limit = capacity if capacity is not None else queue_capacity()
    kept, spans, depth = _ingest(frames, limit, chunk_seconds())
    provider = grounding if grounding is not None else _load_grounding(active_grounding())
    seed = selector_config()
    write_profiles(
        mission_id,
        kept or frames[:1],
        prompts,
        target,
        grounding=provider,
        masker=_load_mask(active_mask()),
        profiles=("fast",),
        selector=seed,
        tracker_config=TrackerConfig(seed=seed.seed),
        created_at=_utc_now(),
        coordinate_frame=CoordinateFrame(name="mission_enu", metric_scale="unavailable"),
    )
    gap_ids = _append_queue_gaps(target, mission_id, spans)
    budget = BudgetController(gpu_slots=slots if slots is not None else gpu_slots())
    pressure = bool(spans)
    degradations = _shed_under_pressure(budget, pressure)
    if prepare_geometry is not None and budget.admit("dense_geometry", pressure=pressure):
        try:
            prepare_geometry(target)
        except Exception:
            logger.warning("dense geometry failed mission=%s; tracks kept", mission_id)
            degradations.append(
                Degradation(
                    code="provider_unavailable",
                    stage="dense_geometry",
                    detail="dense geometry failed; track updates were kept",
                )
            )
        finally:
            budget.release("dense_geometry")
    if fixture_geometry:
        write_track_boxes(target, mission_id)
    reviewer = None
    proposal_provider = None
    if pressure or not budget.admit("vlm", pressure=pressure):
        proposal_provider = UnavailableVlm()
    else:
        budget.release("vlm")
    if pressure or not budget.admit("review", pressure=pressure):
        reviewer = UnavailableReview()
    else:
        budget.release("review")
    graph_regions = (
        list(regions) if regions is not None else ([_REGION] if fixture_geometry else None)
    )
    run_mission_graph(
        mission_id,
        dest=target,
        provider=proposal_provider,
        regions=graph_regions,
    )
    run_mission_verify(mission_id, dest=target, reviewer=reviewer)
    elapsed = time.perf_counter() - started
    media = _media_duration(frames, duration_sec)
    _append_stage(
        target,
        queue_depth=depth,
        skipped=sum(span.skipped_frames for span in spans),
        processed=len(kept),
        elapsed=elapsed,
        degradations=degradations,
        trigger="fast",
    )
    timeline = read_model(target / "timeline.json", SceneTimeline)
    new_ids = publish_accepted(
        target,
        timeline,
        zone_id=zone_id(),
        sensor_id=sensor_id(mission_id),
        published_at=published_at,
    )
    lags = _lags(timeline, media, elapsed, chunk_seconds())
    factor = elapsed / media if media else 0.0
    run = ProfileRun(
        dest=target,
        profile="fast",
        published_event_ids=published_ids(target),
        degradations=[item.code for item in degradations],
        backlog=list(budget.shed),
        queue_depth=depth,
        queue_capacity=limit,
        real_time_factor=factor,
        lags_sec=lags,
        gap_ids=gap_ids,
        replayed=False,
        media_duration_sec=media,
    )
    _write_state(
        target,
        {
            "input_digest": digest,
            "fast_done": True,
            "deep_done": False,
            "published_event_ids": run.published_event_ids,
            "backlog": run.backlog,
            "queue_depth": depth,
            "queue_capacity": limit,
            "real_time_factor": factor,
            "lags_sec": lags,
            "degradations": run.degradations,
            "media_duration_sec": media,
            "new_ids": new_ids,
        },
    )
    validate_bundle(target)
    logger.info(
        "fast profile mission=%s published=%d queue_depth=%d rtf=%.4f",
        mission_id,
        len(run.published_event_ids),
        depth,
        factor,
    )
    return run


def run_deep_profile(
    mission_id: str,
    frames: list[FrameSignal],
    prompts: list[Prompt],
    *,
    dest: Path | None = None,
    grounding: object | None = None,
    fixture_geometry: bool = False,
    duration_sec: float | None = None,
    capacity: int | None = None,
    slots: int | None = None,
    published_at: datetime | None = None,
) -> ProfileRun:
    """Revise a fast run. Fast events stay in the timeline and gain a successor.

    Args:
        mission_id: Mission that already has, or will receive, a fast run.
        frames: The same frames the fast pass used.
        prompts: The same prompts the fast pass used.
        dest: Artifact directory.
        grounding: Keyframe detector for both passes.
        fixture_geometry: Write track boxes before the graph runs.
        duration_sec: Media length forwarded to the fast pass.
        capacity: Queue capacity forwarded to the fast pass.
        slots: GPU slots forwarded to the fast pass.
        published_at: Envelope timestamp for newly accepted deep events.

    Returns:
        A deep result whose manifest names the fast manifest digest.
    """
    target = Path(dest) if dest is not None else analysis_dir(mission_id)
    digest = _input_digest(frames, prompts)
    state = _read_state(target)
    if state.get("deep_done") and state.get("input_digest") == digest:
        return _from_state(target, state, profile="deep", replayed=True)
    fast = run_fast_profile(
        mission_id,
        frames,
        prompts,
        dest=target,
        grounding=grounding,
        fixture_geometry=fixture_geometry,
        duration_sec=duration_sec,
        capacity=capacity,
        slots=slots,
        published_at=published_at,
    )
    if fast.replayed and _manifest(target).profile == "deep":
        return _from_state(target, _read_state(target), profile="deep", replayed=True)
    fast_digest = sha256_bytes((target / "manifest.json").read_bytes())
    fast_events = list(read_model(target / "timeline.json", SceneTimeline).events)
    provider = grounding if grounding is not None else _load_grounding(active_grounding())
    output = run_pass(
        mission_id,
        frames,
        prompts,
        profile="deep",
        grounding=provider,
        masker=_load_mask(active_mask()),
    )
    _append_deep_tracks(target, mission_id, output.observations)
    for gap in _gap_records(mission_id, output):
        AnalysisStore(target).append_gap(gap)
    if fixture_geometry:
        write_track_boxes(target, mission_id)
    _append_deep_graph(target, mission_id, fast_events)
    _mark_deep(target, before_digest=fast_digest, stages=output.stages)
    timeline = read_model(target / "timeline.json", SceneTimeline)
    publish_accepted(
        target,
        timeline,
        zone_id=zone_id(),
        sensor_id=sensor_id(mission_id),
        published_at=published_at,
    )
    validate_bundle(target)
    state = _read_state(target)
    state["deep_done"] = True
    state["published_event_ids"] = published_ids(target)
    state["supersedes_sha256"] = fast_digest
    _write_state(target, state)
    logger.info(
        "deep profile mission=%s supersedes=%s published=%d",
        mission_id,
        fast_digest,
        len(state["published_event_ids"]),
    )
    return ProfileRun(
        dest=target,
        profile="deep",
        published_event_ids=list(state["published_event_ids"]),
        degradations=list(state.get("degradations") or []),
        backlog=list(state.get("backlog") or []),
        queue_depth=int(state.get("queue_depth") or 0),
        queue_capacity=int(state.get("queue_capacity") or 0),
        real_time_factor=float(state.get("real_time_factor") or fast.real_time_factor),
        lags_sec=list(state.get("lags_sec") or []),
        replayed=False,
        superseded=True,
        media_duration_sec=float(state.get("media_duration_sec") or fast.media_duration_sec),
    )


def read_profile_status(mission_id: str, dest: Path | None = None) -> dict:
    """Return queue, degradation, and publication status for a mission."""
    target = Path(dest) if dest is not None else analysis_dir(mission_id)
    manifest = _manifest(target)
    state = _read_state(target)
    return {
        "mission_id": mission_id,
        "profile": manifest.profile,
        "queue_depth": int(state.get("queue_depth") or _max_queue(manifest)),
        "queue_capacity": int(state.get("queue_capacity") or 0),
        "degradations": [item.model_dump(mode="json") for item in manifest.degradations],
        "stages": [item.model_dump(mode="json") for item in manifest.stages],
        "backlog": list(state.get("backlog") or []),
        "published_event_ids": published_ids(target),
        "supersedes_sha256": manifest.supersedes_sha256,
        "real_time_factor": state.get("real_time_factor"),
    }


def write_track_boxes(dest: Path, mission_id: str) -> None:
    """Write one unavailable-scale box per track so deterministic events can fire."""
    tracks = read_jsonl(dest / "tracks.jsonl", TrackRecord)
    seen: set[str] = set()
    for record in tracks:
        if record.track_id in seen:
            continue
        seen.add(record.track_id)
        millis = int(round(record.t_sec * 1000))
        sample = GeometrySample(
            schema_version=SCHEMA_GEOMETRY,
            mission_id=mission_id,
            sample_id=f"geo-{record.track_id}-{millis:07d}",
            subject_id=record.track_id,
            t_sec=record.t_sec,
            frame="mission_enu",
            center_m=[0.0, 0.0, 0.0],
            extent_m=[0.4, 0.4, 0.4],
            quaternion_xyzw=list(_QUAT),
            metric_scale="unavailable",
        )
        path = dest / "geometry" / record.track_id / f"{millis:07d}.json"
        if not path.is_file():
            write_bytes(path, canonical_bytes(sample))


def frames_from_rows(rows: list[dict]) -> list[FrameSignal]:
    """Build causal frame signals from indexed mission rows."""
    frames = [
        FrameSignal(t_sec=float(row.get("t_sec") or 0.0), embedding=(1.0, 0.0), quality_ok=True)
        for row in rows
    ]
    frames.sort(key=lambda item: item.t_sec)
    return frames


def _ingest(
    frames: list[FrameSignal], capacity: int, chunk_sec: float
) -> tuple[list[FrameSignal], list[DiscardedSpan], int]:
    queue = BoundedFrameQueue(capacity)
    kept: list[FrameSignal] = []
    spans: list[DiscardedSpan] = []
    if not frames:
        return kept, spans, 0
    chunk_end = frames[0].t_sec + chunk_sec
    for frame in frames:
        while frame.t_sec >= chunk_end:
            kept.extend(queue.drain())
            chunk_end += chunk_sec
        spans.extend(queue.push(frame))
    kept.extend(queue.drain())
    return kept, spans, queue.max_depth


def _append_queue_gaps(dest: Path, mission_id: str, spans: list[DiscardedSpan]) -> list[str]:
    store = AnalysisStore(dest)
    ids: list[str] = []
    for index, span in enumerate(spans):
        gap_id = f"gap-queue-{index:04d}"
        store.append_gap(
            GapRecord(
                schema_version=SCHEMA_GAP,
                mission_id=mission_id,
                gap_id=gap_id,
                start_sec=span.start_sec,
                end_sec=span.end_sec,
                reason=span.reason,
                skipped_frames=span.skipped_frames,
            )
        )
        ids.append(gap_id)
    return ids


def _shed_under_pressure(budget: BudgetController, pressure: bool) -> list[Degradation]:
    """Record optional stages that will not run when the queue coalesced or slots are zero."""
    if not pressure and budget.gpu_slots > 0:
        return []
    items: list[Degradation] = []
    for stage in ("dense_geometry", "vlm", "review"):
        if stage not in budget.shed:
            budget.shed.append(stage)
        items.append(
            Degradation(
                code="budget_shed",
                stage=stage,
                detail=f"{stage} shed so track continuity keeps the queue",
            )
        )
    return items


def _append_stage(
    dest: Path,
    *,
    queue_depth: int,
    skipped: int,
    processed: int,
    elapsed: float,
    degradations: list[Degradation],
    trigger: str,
) -> None:
    path = dest / "manifest.json"
    manifest = read_model(path, AnalysisManifest)
    previous = sha256_bytes(path.read_bytes())
    stage = StageReport(
        stage="profile_orchestrator",
        queue_delay_sec=0.0,
        inference_time_sec=elapsed,
        processed_frames=processed,
        skipped_frames=skipped,
        trigger_reason=trigger,
        degradation_flags=[item.code for item in degradations],
        queue_depth=queue_depth,
    )
    updated = manifest.model_copy(
        update={
            "artifacts": _refresh_artifacts(dest, list(manifest.artifacts)),
            "degradations": _merge_degradations(list(manifest.degradations), degradations),
            "stages": [*manifest.stages, stage],
            "supersedes_sha256": previous,
        }
    )
    AnalysisStore(dest).write_manifest(updated)


def _append_deep_tracks(dest: Path, mission_id: str, observations: list[Observation]) -> None:
    existing = read_jsonl(dest / "tracks.jsonl", TrackRecord)
    fast_at = {(round(row.t_sec, 6), row.label_normalized): row.observation_id for row in existing}
    known = {row.track_id for row in existing}
    known.update(observation.track_id for observation in observations)
    store = AnalysisStore(dest)
    for index, observation in enumerate(observations):
        link = None
        if (
            observation.identity_link_id
            and observation.identity_reason
            and observation.identity_link_id in known
        ):
            link = IdentityLink(
                linked_track_id=observation.identity_link_id,
                reason=observation.identity_reason,
            )
        supersedes = fast_at.get((round(observation.t_sec, 6), observation.label_normalized))
        record = TrackRecord(
            schema_version=SCHEMA_TRACK,
            mission_id=mission_id,
            observation_id=f"obs-deep-{index:04d}",
            track_id=observation.track_id,
            state=observation.state,  # type: ignore[arg-type]
            t_sec=observation.t_sec,
            prompt_id=observation.prompt_id,
            label_raw=observation.label_raw,
            label_normalized=observation.label_normalized,
            box=Box2D(xywh_norm=list(observation.xywh)),
            confidence=observation.confidence,
            identity_link=link,
            supersedes=supersedes,
        )
        store.append_track(record)
        known.add(observation.track_id)


def _append_deep_graph(dest: Path, mission_id: str, fast_events: list[TimelineEvent]) -> None:
    tracks = read_jsonl(dest / "tracks.jsonl", TrackRecord)
    samples = _load_geometry(dest)
    manifest = _manifest(dest)
    reduced = reduce_graph(
        mission_id,
        tracks,
        samples,
        frame_name=manifest.coordinate_frame.name,
        metric=manifest.coordinate_frame.metric_scale == "metric",
        region_ids={_REGION.node_id},
    )
    existing_deltas = {row.delta_id for row in read_jsonl(dest / "graph-deltas.jsonl", GraphDelta)}
    existing_events = {event.event_id for event in fast_events}
    fresh = [delta for delta in reduced.deltas if delta.delta_id not in existing_deltas]
    fresh_ids = existing_deltas | {delta.delta_id for delta in fresh}
    store = AnalysisStore(dest)
    for delta in fresh:
        store.append_delta(delta)
    linked: list[TimelineEvent] = []
    for event in reduced.events:
        if event.event_id in existing_events:
            continue
        if any(ref not in fresh_ids for ref in event.state_delta_refs):
            continue
        match = next((item.event_id for item in fast_events if item.type == event.type), None)
        linked.append(event.model_copy(update={"supersedes": match}) if match else event)
    if not linked:
        _retarget_timeline(dest, fast_events, profile="deep")
        return
    _retarget_timeline(dest, [*fast_events, *linked], profile="deep")


def _retarget_timeline(dest: Path, events: list[TimelineEvent], *, profile: str) -> None:
    path = dest / "timeline.json"
    timeline = read_model(path, SceneTimeline)
    previous = sha256_bytes(path.read_bytes())
    updated = timeline.model_copy(
        update={
            "profile": profile,
            "events": events,
            "supersedes_sha256": previous,
        }
    )
    AnalysisStore(dest).write_timeline(updated)


def _mark_deep(dest: Path, *, before_digest: str, stages: list[StageReport]) -> None:
    path = dest / "manifest.json"
    manifest = read_model(path, AnalysisManifest)
    updated = manifest.model_copy(
        update={
            "profile": "deep",
            "artifacts": _refresh_artifacts(dest, list(manifest.artifacts)),
            "stages": [*manifest.stages, *stages],
            "supersedes_sha256": before_digest,
        }
    )
    AnalysisStore(dest).write_manifest(updated)


def _lags(timeline: SceneTimeline, media: float, elapsed: float, chunk_sec: float) -> list[float]:
    types = fast_event_types()
    chunks = max(1, math.ceil(media / chunk_sec)) if media else 1
    share = elapsed / chunks
    return [
        share
        for event in timeline.events
        if event.verification.status == "accepted" and event.type in types
    ]


def _media_duration(frames: list[FrameSignal], duration_sec: float | None) -> float:
    if duration_sec is not None and duration_sec > 0:
        return duration_sec
    if len(frames) < 2:
        return 1.0
    return max(1e-3, frames[-1].t_sec - frames[0].t_sec)


def _input_digest(frames: list[FrameSignal], prompts: list[Prompt]) -> str:
    payload = json.dumps(
        {
            "t": [frame.t_sec for frame in frames],
            "q": [frame.quality_ok for frame in frames],
            "p": [prompt.prompt_id for prompt in prompts],
        },
        separators=(",", ":"),
    )
    return sha256_bytes(payload.encode("utf-8"))


def _read_state(dest: Path) -> dict:
    path = dest / STATE_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(dest: Path, state: dict) -> None:
    path = dest / STATE_NAME
    path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")


def _from_state(dest: Path, state: dict, *, profile: str, replayed: bool) -> ProfileRun:
    return ProfileRun(
        dest=dest,
        profile=profile,
        published_event_ids=list(state.get("published_event_ids") or published_ids(dest)),
        degradations=list(state.get("degradations") or []),
        backlog=list(state.get("backlog") or []),
        queue_depth=int(state.get("queue_depth") or 0),
        queue_capacity=int(state.get("queue_capacity") or 0),
        real_time_factor=float(state.get("real_time_factor") or 0.0),
        lags_sec=list(state.get("lags_sec") or []),
        replayed=replayed,
        superseded=bool(state.get("deep_done")),
        media_duration_sec=float(state.get("media_duration_sec") or 0.0),
    )


def _manifest(dest: Path) -> AnalysisManifest:
    return read_model(dest / "manifest.json", AnalysisManifest)


def _max_queue(manifest: AnalysisManifest) -> int:
    return max((stage.queue_depth for stage in manifest.stages), default=0)


def _merge_degradations(current: list[Degradation], extra: list[Degradation]) -> list[Degradation]:
    seen = {(item.code, item.stage, item.detail) for item in current}
    merged = list(current)
    for item in extra:
        key = (item.code, item.stage, item.detail)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _refresh_artifacts(dest: Path, artifacts: list[ArtifactEntry]) -> list[ArtifactEntry]:
    refreshed: list[ArtifactEntry] = []
    known: set[str] = set()
    for entry in artifacts:
        file_path = dest / entry.path
        known.add(entry.path)
        if not file_path.is_file():
            refreshed.append(entry)
            continue
        payload = file_path.read_bytes()
        refreshed.append(
            entry.model_copy(update={"sha256": sha256_bytes(payload), "bytes": len(payload)})
        )
    geometry = dest / "geometry"
    if geometry.is_dir():
        for file_path in sorted(geometry.rglob("*.json")):
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
            known.add(rel)
    return refreshed


def _load_geometry(dest: Path) -> list[GeometrySample]:
    directory = dest / "geometry"
    if not directory.is_dir():
        return []
    return [read_model(path, GeometrySample) for path in sorted(directory.rglob("*.json"))]


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
