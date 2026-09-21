"""Fast causal and deep forward/backward track passes.

The fast profile selects keyframes inside bounded chunks and associates
tracks causally. The deep profile reselects keyframes globally, runs the same
association forward, then links identities backward across long gaps and
cuts. Both profiles write the same track contract. A later profile's
observations name the earlier observation they revise.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from selfsuvis.pipeline.analysis4d.io import (
    canonical_bytes,
    canonical_line,
    sha256_bytes,
    write_bytes,
)
from selfsuvis.pipeline.analysis4d.keyframes import (
    FrameSignal,
    KeyframePick,
    SelectorConfig,
    SkipSpan,
    select_keyframes,
)
from selfsuvis.pipeline.analysis4d.providers import (
    CountProvider,
    GroundingProvider,
    KinematicMask,
    MaskProvider,
    Prompt,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_DELTA,
    SCHEMA_GAP,
    SCHEMA_MASK,
    SCHEMA_PROPOSAL,
    SCHEMA_QA,
    SCHEMA_TIMELINE,
    SCHEMA_TRACK,
    SCHEMA_TRACK_AUDIT,
    AnalysisManifest,
    ArtifactEntry,
    Box2D,
    CoordinateFrame,
    CountDisagreementNote,
    Degradation,
    GapRecord,
    IdentityLink,
    KeyframeNote,
    MaskArtifact,
    MemoryResetNote,
    ModelProvenance,
    SceneTimeline,
    StageReport,
    TrackAudit,
    TrackRecord,
)
from selfsuvis.pipeline.analysis4d.tracks import (
    Detection,
    MemoryReset,
    Observation,
    Tracker,
    TrackerConfig,
)

_MASK = 8


@dataclass
class PassOutput:
    """In-memory result of one profile before files are written."""

    profile: str
    observations: list[Observation] = field(default_factory=list)
    picks: list[KeyframePick] = field(default_factory=list)
    resets: list[MemoryReset] = field(default_factory=list)
    pending_counts: list[tuple[float, str, str, int]] = field(default_factory=list)
    skips: list[SkipSpan] = field(default_factory=list)
    stages: list[StageReport] = field(default_factory=list)
    elapsed_sec: float = 0.0


def run_pass(
    mission_id: str,
    frames: list[FrameSignal],
    prompts: list[Prompt],
    *,
    profile: str,
    grounding: GroundingProvider,
    counter: CountProvider | None = None,
    masker: MaskProvider | None = None,
    selector: SelectorConfig | None = None,
    tracker_config: TrackerConfig | None = None,
) -> PassOutput:
    """Run one profile. ``deep`` adds identity links after the forward pass.

    Args:
        mission_id: Mission whose tracks are being built.
        frames: Timestamped frames. Quality failures become skip spans.
        prompts: Text or exemplar prompts passed to the providers.
        profile: ``fast`` or ``deep``.
        grounding: Keyframe detector. Count-only hits are ignored.
        counter: Optional count expert. Disagreement does not create tracks.
        masker: Mask refiner. Defaults to kinematic propagation.
        selector: Keyframe budget and gates.
        tracker_config: Association and occlusion thresholds.

    Returns:
        Observations, keyframes, resets, disagreements, and stage telemetry.
    """
    started = time.perf_counter()
    cfg = selector or SelectorConfig()
    mask = masker or KinematicMask()
    select_started = time.perf_counter()
    picks, skips = select_keyframes(frames, cfg, profile=profile)
    select_elapsed = time.perf_counter() - select_started
    ground_at = {_stamp(pick.t_sec) for pick in picks}
    notes = {_stamp(pick.t_sec): pick for pick in picks}
    tracker = Tracker(
        mission_id=mission_id,
        profile=profile,
        config=tracker_config or TrackerConfig(seed=cfg.seed),
        id_prefix=f"{profile}-",
    )
    usable = [frame for frame in sorted(frames, key=lambda item: item.t_sec) if frame.quality_ok]
    ground_elapsed = 0.0
    grounded = 0
    pending_counts: list[tuple[float, str, str, int]] = []
    for frame in usable:
        stamp = _stamp(frame.t_sec)
        if frame.scene_cut or frame.calibration_changed:
            mask.reset("scene_cut" if frame.scene_cut else "calibration_change")
        detections: list[Detection] = []
        if stamp in ground_at:
            t0 = time.perf_counter()
            detections = list(mask.refine(frame, list(grounding.ground(frame, prompts))))
            ground_elapsed += time.perf_counter() - t0
            grounded += 1
            if counter is not None:
                for hit in counter.count(frame, prompts):
                    pending_counts.append(
                        (frame.t_sec, hit.prompt_id, hit.label_normalized, hit.count)
                    )
        extra = tracker.step(frame, detections)
        for reason in dict.fromkeys(extra):
            current = notes.get(stamp)
            if current is None:
                notes[stamp] = KeyframePick(t_sec=frame.t_sec, reasons=(reason,), forced=True)
            else:
                notes[stamp] = current.with_reason(reason)
    if profile == "deep":
        tracker.link_identities()
    ordered = sorted(notes.values(), key=lambda pick: pick.t_sec)
    output = PassOutput(
        profile=profile,
        observations=list(tracker.observations),
        picks=ordered,
        resets=list(tracker.resets),
        pending_counts=pending_counts,
        skips=skips,
        elapsed_sec=time.perf_counter() - started,
    )
    output.stages = _stages(
        profile,
        picks=ordered,
        processed=len(usable),
        grounded=grounded,
        skips=skips,
        select_elapsed=select_elapsed,
        ground_elapsed=ground_elapsed,
        total_elapsed=output.elapsed_sec,
        model=getattr(grounding, "provenance", None),
    )
    return output


def write_profiles(
    mission_id: str,
    frames: list[FrameSignal],
    prompts: list[Prompt],
    dest: Path,
    *,
    grounding: GroundingProvider,
    counter: CountProvider | None = None,
    masker: MaskProvider | None = None,
    profiles: tuple[str, ...] = ("fast", "deep"),
    selector: SelectorConfig | None = None,
    tracker_config: TrackerConfig | None = None,
    created_at: str = "2026-01-15T00:00:00Z",
    coordinate_frame: CoordinateFrame | None = None,
) -> Path:
    """Run the requested profiles and write one validated artifact directory.

    Args:
        mission_id: Mission id stored on every record.
        frames: Decoded frames for every profile.
        prompts: Prompts shared by the profiles.
        dest: Empty directory that will receive the bundle.
        grounding: Keyframe detector.
        counter: Optional count expert.
        masker: Optional mask refiner.
        profiles: ``fast``, ``deep``, or both. Both are written in time order.
        selector: Shared selector config. The seed is stored on the audit.
        tracker_config: Shared tracker config.
        created_at: UTC timestamp for the manifest.
        coordinate_frame: Mission frame. Defaults to unavailable scale.

    Returns:
        ``dest`` after ``manifest.json`` is written.

    Raises:
        FileExistsError: ``dest`` already contains a manifest.
    """
    dest = Path(dest)
    if (dest / "manifest.json").is_file():
        raise FileExistsError(dest / "manifest.json")
    dest.mkdir(parents=True, exist_ok=True)
    cfg = selector or SelectorConfig()
    passes = [
        run_pass(
            mission_id,
            frames,
            prompts,
            profile=profile,
            grounding=grounding,
            counter=counter,
            masker=masker,
            selector=cfg,
            tracker_config=tracker_config,
        )
        for profile in profiles
    ]
    stored = passes[-1]
    records = _records(mission_id, passes)
    mask_files = _write_masks(dest, mission_id, records)
    _write_tracks(dest, records)
    gaps = _gap_records(mission_id, stored)
    if gaps:
        write_bytes(dest / "gaps.jsonl", b"".join(canonical_line(gap) for gap in gaps))
    else:
        write_bytes(dest / "gaps.jsonl", b"")
    for name in ("graph-deltas.jsonl", "proposals.jsonl", "qa.jsonl"):
        write_bytes(dest / name, b"")
    frame = coordinate_frame or CoordinateFrame(name="mission_enu", metric_scale="unavailable")
    degradations = _degradations(grounding, records, stored)
    timeline = SceneTimeline(
        schema_version=SCHEMA_TIMELINE,
        mission_id=mission_id,
        profile=stored.profile,
        coordinate_frame=frame,
        model_manifest_ref="manifest.json",
        degradations=degradations,
        events=[],
        qa_pairs=[],
    )
    write_bytes(dest / "timeline.json", canonical_bytes(timeline))
    audit = _audit(mission_id, stored, passes, grounding, counter, masker, cfg.seed)
    write_bytes(dest / "track-audit.json", canonical_bytes(audit))
    stages = _stored_stages(passes)
    models = _models(passes)
    artifacts = _artifact_entries(dest, mask_files, gaps)
    manifest = AnalysisManifest(
        schema_version="ss-video.analysis4d-manifest.v1",
        mission_id=mission_id,
        profile=stored.profile,
        coordinate_frame=frame,
        created_at=created_at,
        artifacts=artifacts,
        models=models,
        degradations=degradations,
        stages=stages,
        keyframes_sec=[pick.t_sec for pick in stored.picks],
    )
    write_bytes(dest / "manifest.json", canonical_bytes(manifest))
    return dest


def _stages(
    profile: str,
    *,
    picks: list[KeyframePick],
    processed: int,
    grounded: int,
    skips: list[SkipSpan],
    select_elapsed: float,
    ground_elapsed: float,
    total_elapsed: float,
    model: ModelProvenance | None,
) -> list[StageReport]:
    skipped = sum(span.skipped_frames for span in skips)
    trigger = _trigger(picks)
    return [
        StageReport(
            stage=f"{profile}_selector",
            queue_delay_sec=0.0,
            inference_time_sec=select_elapsed,
            processed_frames=processed,
            skipped_frames=skipped,
            trigger_reason=trigger,
            model=None,
            degradation_flags=[],
            queue_depth=0,
        ),
        StageReport(
            stage=f"{profile}_grounding",
            queue_delay_sec=0.0,
            inference_time_sec=ground_elapsed,
            processed_frames=grounded,
            skipped_frames=0,
            trigger_reason=trigger,
            model=model,
            degradation_flags=[],
            queue_depth=0,
        ),
        StageReport(
            stage=f"{profile}_tracker",
            queue_delay_sec=0.0,
            inference_time_sec=max(0.0, total_elapsed - select_elapsed - ground_elapsed),
            processed_frames=processed,
            skipped_frames=0,
            trigger_reason=trigger,
            model=None,
            degradation_flags=[],
            queue_depth=0,
        ),
    ]


def _trigger(picks: list[KeyframePick]) -> str:
    reasons: list[str] = []
    for pick in picks:
        for reason in pick.reasons:
            if reason not in reasons:
                reasons.append(reason)
    return "+".join(reasons)[:180] if reasons else "empty"


def _records(mission_id: str, passes: list[PassOutput]) -> list[TrackRecord]:
    built: list[tuple[float, int, TrackRecord]] = []
    fast_at: dict[tuple[float, str], str] = {}
    for profile_index, output in enumerate(passes):
        for index, observation in enumerate(output.observations):
            observation_id = f"obs-{output.profile}-{index:04d}"
            if output.profile == "fast":
                fast_at[_stamp(observation.t_sec), observation.label_normalized] = observation_id
            supersedes = None
            if output.profile == "deep":
                supersedes = fast_at.get((_stamp(observation.t_sec), observation.label_normalized))
            link = None
            if observation.identity_link_id and observation.identity_reason:
                link = IdentityLink(
                    linked_track_id=observation.identity_link_id,
                    reason=observation.identity_reason,
                )
            record = TrackRecord(
                schema_version=SCHEMA_TRACK,
                mission_id=mission_id,
                observation_id=observation_id,
                track_id=observation.track_id,
                state=observation.state,
                t_sec=observation.t_sec,
                prompt_id=observation.prompt_id,
                label_raw=observation.label_raw,
                label_normalized=observation.label_normalized,
                box=Box2D(xywh_norm=list(observation.xywh)),
                mask_ref=_mask_path(observation.track_id, observation.t_sec),
                confidence=observation.confidence,
                negative_prompts=list(observation.negative_prompts),
                exemplars=list(observation.exemplars),
                model=None,
                identity_link=link,
                supersedes=supersedes,
            )
            built.append((observation.t_sec, profile_index, record))
    built.sort(key=lambda item: (item[0], item[1], item[2].observation_id))
    return [record for _t, _index, record in built]


def _write_tracks(dest: Path, records: list[TrackRecord]) -> None:
    payload = b"".join(canonical_line(record) for record in records)
    write_bytes(dest / "tracks.jsonl", payload)


def _write_masks(
    dest: Path, mission_id: str, records: list[TrackRecord]
) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for record in records:
        path = record.mask_ref or _mask_path(record.track_id, record.t_sec)
        mask = MaskArtifact(
            schema_version=SCHEMA_MASK,
            mission_id=mission_id,
            track_id=record.track_id,
            t_sec=record.t_sec,
            width=_MASK,
            height=_MASK,
            bits=_raster(tuple(record.box.xywh_norm)),
        )
        payload = canonical_bytes(mask)
        write_bytes(dest / path, payload)
        files.append((path, payload))
    return files


def _gap_records(mission_id: str, output: PassOutput) -> list[GapRecord]:
    gaps: list[GapRecord] = []
    for index, span in enumerate(output.skips):
        gaps.append(
            GapRecord(
                schema_version=SCHEMA_GAP,
                mission_id=mission_id,
                gap_id=f"gap-{output.profile}-{index:04d}",
                start_sec=span.start_sec,
                end_sec=span.end_sec,
                reason=span.reason,
                skipped_frames=span.skipped_frames,
            )
        )
    return gaps


def _degradations(
    grounding: GroundingProvider, records: list[TrackRecord], output: PassOutput
) -> list[Degradation]:
    del output
    items: list[Degradation] = []
    if not records:
        items.append(
            Degradation(code="empty_scene", stage="tracker", detail="no track observations")
        )
    if getattr(grounding, "provider_id", "") == "unavailable" or getattr(
        grounding, "failed", False
    ):
        items.append(
            Degradation(
                code="provider_unavailable",
                stage="grounding",
                detail="pinned grounding provider did not load",
            )
        )
    return items


def _audit(
    mission_id: str,
    stored: PassOutput,
    passes: list[PassOutput],
    grounding: GroundingProvider,
    counter: CountProvider | None,
    masker: MaskProvider | None,
    seed: int,
) -> TrackAudit:
    resets: list[MemoryResetNote] = []
    disagreements: list[CountDisagreementNote] = []
    for output in passes:
        resets.extend(
            MemoryResetNote(t_sec=item.t_sec, reason=item.reason, scope=item.scope)
            for item in output.resets
        )
        disagreements.extend(_resolve_disagreements(output))
    notes: list[KeyframeNote] = []
    for output in passes:
        for pick in output.picks:
            notes.append(
                KeyframeNote(
                    t_sec=pick.t_sec,
                    reasons=list(pick.reasons),
                    forced=pick.forced,
                    profile=output.profile,
                )
            )
    return TrackAudit(
        schema_version=SCHEMA_TRACK_AUDIT,
        mission_id=mission_id,
        profile=stored.profile,
        seed=seed,
        grounding_provider=getattr(grounding, "provider_id", "unknown"),
        mask_provider=getattr(masker or KinematicMask(), "provider_id", "kinematic"),
        count_provider="" if counter is None else getattr(counter, "provider_id", ""),
        keyframes=notes,
        memory_resets=resets,
        count_disagreements=disagreements,
    )


def _resolve_disagreements(output: PassOutput) -> list[CountDisagreementNote]:
    """Fill track totals after association and keep only real disagreements."""
    resolved: list[CountDisagreementNote] = []
    for t_sec, prompt_id, label, provider_count in output.pending_counts:
        total = sum(
            1
            for observation in output.observations
            if observation.label_normalized == label
            and _stamp(observation.t_sec) == _stamp(t_sec)
            and observation.state in {"tentative", "confirmed", "occluded"}
        )
        if total == provider_count:
            continue
        resolved.append(
            CountDisagreementNote(
                t_sec=t_sec,
                prompt_id=prompt_id,
                label_normalized=label,
                track_count=total,
                provider_count=provider_count,
            )
        )
        _stamp_reason(output, t_sec, "count_disagreement")
    return resolved


def _stamp_reason(output: PassOutput, t_sec: float, reason: str) -> None:
    for index, pick in enumerate(output.picks):
        if _stamp(pick.t_sec) == _stamp(t_sec):
            output.picks[index] = pick.with_reason(reason)
            return
    output.picks.append(KeyframePick(t_sec=t_sec, reasons=(reason,), forced=True))
    output.picks.sort(key=lambda pick: pick.t_sec)


def _stored_stages(passes: list[PassOutput]) -> list[StageReport]:
    """Keep skip totals on the stored profile only, so gap records match once.

    Count disagreements are stamped after the pass builds stages, so the
    trigger string is refreshed from the final keyframe notes.
    """
    stored = passes[-1].profile
    stages: list[StageReport] = []
    for output in passes:
        trigger = _trigger(output.picks)
        for stage in output.stages:
            updates: dict[str, object] = {"trigger_reason": trigger}
            if output.profile != stored and stage.skipped_frames:
                updates["skipped_frames"] = 0
            stages.append(stage.model_copy(update=updates))
    if not stages:
        stages.append(
            StageReport(
                stage="selector",
                queue_delay_sec=0.0,
                inference_time_sec=0.0,
                processed_frames=0,
                skipped_frames=0,
                trigger_reason="empty",
            )
        )
    return stages


def _models(passes: list[PassOutput]) -> list[ModelProvenance]:
    found: list[ModelProvenance] = []
    seen: set[str] = set()
    for output in passes:
        for stage in output.stages:
            if stage.model is None or stage.model.model_id in seen:
                continue
            seen.add(stage.model.model_id)
            found.append(stage.model)
    return found


def _artifact_entries(
    dest: Path, mask_files: list[tuple[str, bytes]], gaps: list[GapRecord]
) -> list[ArtifactEntry]:
    named = [
        ("tracks.jsonl", "tracks", SCHEMA_TRACK),
        ("graph-deltas.jsonl", "graph_deltas", SCHEMA_DELTA),
        ("proposals.jsonl", "proposals", SCHEMA_PROPOSAL),
        ("timeline.json", "timeline", SCHEMA_TIMELINE),
        ("qa.jsonl", "qa", SCHEMA_QA),
        ("track-audit.json", "track_audit", SCHEMA_TRACK_AUDIT),
    ]
    if gaps:
        named.append(("gaps.jsonl", "gaps", SCHEMA_GAP))
    entries: list[ArtifactEntry] = []
    for path, kind, schema in named:
        payload = (dest / path).read_bytes()
        entries.append(
            ArtifactEntry(
                path=path,
                kind=kind,
                schema_version=schema,
                sha256=sha256_bytes(payload),
                bytes=len(payload),
            )
        )
    for path, payload in mask_files:
        entries.append(
            ArtifactEntry(
                path=path,
                kind="masks",
                schema_version=SCHEMA_MASK,
                sha256=sha256_bytes(payload),
                bytes=len(payload),
            )
        )
    return entries


def _raster(xywh: tuple[float, float, float, float]) -> str:
    x, y, width, height = xywh
    chars: list[str] = []
    for row in range(_MASK):
        for col in range(_MASK):
            cx = (col + 0.5) / _MASK
            cy = (row + 0.5) / _MASK
            inside = x <= cx < x + width and y <= cy < y + height
            chars.append("1" if inside else "0")
    return "".join(chars)


def _mask_path(track_id: str, t_sec: float) -> str:
    return f"masks/{track_id}/{int(round(t_sec * 1000)):07d}.json"


def _stamp(t_sec: float) -> float:
    return round(t_sec, 6)
