"""Referential and geometric checks over a 4D artifact directory."""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from selfsuvis.pipeline.analysis4d.geometry import relation_holds, velocity_feasible
from selfsuvis.pipeline.analysis4d.io import read_model, sha256_bytes
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_DELTA,
    SCHEMA_GAP,
    SCHEMA_GEOMETRY,
    SCHEMA_MASK,
    SCHEMA_PROPOSAL,
    SCHEMA_QA,
    SCHEMA_TIMELINE,
    SCHEMA_TRACK,
    AnalysisManifest,
    GapRecord,
    GeometrySample,
    GraphDelta,
    MaskArtifact,
    MissionTruth,
    Proposal,
    QaRecord,
    SceneTimeline,
    TrackRecord,
)

_REQUIRED_KINDS = {
    "tracks": "tracks.jsonl",
    "graph_deltas": "graph-deltas.jsonl",
    "proposals": "proposals.jsonl",
    "timeline": "timeline.json",
    "qa": "qa.jsonl",
}
_SCHEMA_BY_KIND = {
    "tracks": SCHEMA_TRACK,
    "graph_deltas": SCHEMA_DELTA,
    "proposals": SCHEMA_PROPOSAL,
    "timeline": SCHEMA_TIMELINE,
    "qa": SCHEMA_QA,
    "gaps": SCHEMA_GAP,
    "geometry": SCHEMA_GEOMETRY,
    "masks": SCHEMA_MASK,
}
_METRIC_PREDICATES = frozenset({"distance_band", "supports", "contacts"})
_ISSUE_CODES = ("unversioned", "invalid_interval", "mixed_coordinate_frame", "dangling_id")


class ContractError(ValueError):
    """One or more contract violations. ``issues`` is the machine-readable list."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


def issues_from_validation(exc: ValidationError) -> list[str]:
    """Map a pydantic error onto contract issue codes."""
    issues: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        message = str(err["msg"])
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        if any(code in message for code in _ISSUE_CODES):
            issues.append(message if loc in message else f"{message} ({loc})")
            continue
        if "schema_version" in loc:
            issues.append(f"unversioned: {loc}: {message}")
            continue
        if err["type"] == "extra_forbidden":
            issues.append(f"unknown_field: {loc}: {message}")
            continue
        if "end_sec" in loc or "interval" in loc:
            issues.append(f"invalid_interval: {loc}: {message}")
            continue
        issues.append(f"schema: {loc}: {message}")
    return issues


@dataclass
class Bundle:
    """Validated artifact set for one mission."""

    root: Path
    manifest: AnalysisManifest
    tracks: list[TrackRecord]
    deltas: list[GraphDelta]
    proposals: list[Proposal]
    timeline: SceneTimeline
    qa: list[QaRecord]
    gaps: list[GapRecord] = field(default_factory=list)
    geometry: list[GeometrySample] = field(default_factory=list)
    masks: list[MaskArtifact] = field(default_factory=list)
    truth: MissionTruth | None = None


def _parse(path: Path, model_type: type, issues: list[str]):
    try:
        return read_model(path, model_type)
    except ValidationError as exc:
        issues.extend(f"{path.name}: {item}" for item in issues_from_validation(exc))
    except OSError as exc:
        issues.append(f"schema: {path.name}: {exc}")
    return None


def _parse_jsonl(path: Path, model_type: type, issues: list[str]) -> list:
    if not path.is_file():
        issues.append(f"schema: missing {path.name}")
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(model_type.model_validate_json(text))
        except ValidationError as exc:
            prefix = f"{path.name}:{line_number}"
            issues.extend(f"{prefix}: {item}" for item in issues_from_validation(exc))
    return rows


def _unique(ids: list[str], label: str, issues: list[str]) -> set[str]:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            issues.append(f"schema: duplicate {label} {item}")
        seen.add(item)
    return seen


def _require(item: str | None, known: set[str], where: str, issues: list[str]) -> None:
    if item is not None and item not in known:
        issues.append(f"dangling_id: {where} {item}")


def _supersession_cycles(
    pairs: list[tuple[str, str | None]], label: str, issues: list[str]
) -> None:
    graph = {item_id: target for item_id, target in pairs if target}
    for start in graph:
        seen: list[str] = []
        cursor: str | None = start
        while cursor is not None:
            if cursor in seen:
                issues.append(f"schema: supersession cycle in {label}: {' -> '.join(seen)}")
                break
            seen.append(cursor)
            cursor = graph.get(cursor)


def _geometry_index(samples: list[GeometrySample]) -> dict[str, list[GeometrySample]]:
    grouped: dict[str, list[GeometrySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.subject_id, []).append(sample)
    for rows in grouped.values():
        rows.sort(key=lambda sample: sample.t_sec)
    return grouped


def _sample_in_interval(
    samples: list[GeometrySample], start_sec: float, end_sec: float
) -> GeometrySample | None:
    inside = [sample for sample in samples if start_sec <= sample.t_sec < end_sec]
    return inside[-1] if inside else None


def _motion_m(samples: list[GeometrySample], start_sec: float, end_sec: float) -> float | None:
    window = [sample for sample in samples if start_sec <= sample.t_sec < end_sec]
    if len(window) < 2:
        return None
    first, last = window[0].center_m, window[-1].center_m
    return sum((left - right) ** 2 for left, right in zip(first, last, strict=True)) ** 0.5


def validate_bundle(root: Path, *, require_truth: bool = False) -> Bundle:
    """Load and check one mission directory.

    Args:
        root: Directory that contains ``manifest.json`` and the artifact files.
        require_truth: When true, ``truth.json`` must validate. The benchmark sets this.

    Returns:
        The parsed bundle.

    Raises:
        ContractError: Dangling ids, mixed frames, invalid intervals, unversioned
            output, contradictory accepted geometry, or a skipped interval with no gap.
    """
    issues: list[str] = []
    root = Path(root)
    manifest = _parse(root / "manifest.json", AnalysisManifest, issues)
    tracks = _parse_jsonl(root / "tracks.jsonl", TrackRecord, issues)
    deltas = _parse_jsonl(root / "graph-deltas.jsonl", GraphDelta, issues)
    proposals = _parse_jsonl(root / "proposals.jsonl", Proposal, issues)
    timeline = _parse(root / "timeline.json", SceneTimeline, issues)
    qa = _parse_jsonl(root / "qa.jsonl", QaRecord, issues)
    gaps = (
        _parse_jsonl(root / "gaps.jsonl", GapRecord, issues)
        if (root / "gaps.jsonl").is_file()
        else []
    )
    geometry = _load_tree(root / "geometry", GeometrySample, issues)
    masks = _load_tree(root / "masks", MaskArtifact, issues)
    truth = None
    if (root / "truth.json").is_file():
        truth = _parse(root / "truth.json", MissionTruth, issues)
    elif require_truth:
        issues.append("schema: missing truth.json")

    if manifest is None or timeline is None:
        raise ContractError(issues or ["schema: manifest and timeline are required"])

    _check_manifest_hashes(root, manifest, issues)
    _check_identity(manifest, timeline, tracks, deltas, proposals, qa, gaps, geometry, issues)
    _check_references(timeline, tracks, deltas, proposals, qa, geometry, masks, root, issues)
    _check_geometry(manifest, deltas, geometry, issues)
    _check_gaps(manifest, gaps, issues)
    if issues:
        raise ContractError(issues)
    return Bundle(
        root=root,
        manifest=manifest,
        tracks=tracks,
        deltas=deltas,
        proposals=proposals,
        timeline=timeline,
        qa=qa,
        gaps=gaps,
        geometry=geometry,
        masks=masks,
        truth=truth,
    )


def _load_tree(directory: Path, model_type: type, issues: list[str]) -> list:
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.rglob("*.json")):
        parsed = _parse(path, model_type, issues)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _check_manifest_hashes(root: Path, manifest: AnalysisManifest, issues: list[str]) -> None:
    for kind, name in _REQUIRED_KINDS.items():
        match = [entry for entry in manifest.artifacts if entry.kind == kind]
        if len(match) != 1 or match[0].path != name:
            issues.append(f"schema: manifest must list {name} as kind {kind}")
    for entry in manifest.artifacts:
        path = root / entry.path
        if ".." in Path(entry.path).parts or Path(entry.path).is_absolute():
            issues.append(f"schema: artifact path escapes the mission directory: {entry.path}")
            continue
        if not path.is_file():
            issues.append(f"dangling_id: manifest artifact {entry.path}")
            continue
        payload = path.read_bytes()
        if sha256_bytes(payload) != entry.sha256 or len(payload) != entry.bytes:
            issues.append(f"schema: manifest hash mismatch for {entry.path}")
        expected = _SCHEMA_BY_KIND.get(entry.kind)
        if expected is not None and entry.schema_version != expected:
            issues.append(f"unversioned: {entry.path} schema_version {entry.schema_version}")


def _check_identity(
    manifest: AnalysisManifest,
    timeline: SceneTimeline,
    tracks: list[TrackRecord],
    deltas: list[GraphDelta],
    proposals: list[Proposal],
    qa: list[QaRecord],
    gaps: list[GapRecord],
    geometry: list[GeometrySample],
    issues: list[str],
) -> None:
    mission_id = manifest.mission_id
    frame = manifest.coordinate_frame.name
    labeled = (
        ("track", tracks),
        ("delta", deltas),
        ("proposal", proposals),
        ("qa", qa),
        ("gap", gaps),
        ("geometry", geometry),
    )
    for label, rows in labeled:
        for row in rows:
            if row.mission_id != mission_id:
                issues.append(f"schema: {label} mission_id {row.mission_id} != {mission_id}")
    if timeline.mission_id != mission_id:
        issues.append("schema: timeline mission_id differs from the manifest")
    if timeline.profile != manifest.profile:
        issues.append("schema: timeline profile differs from the manifest")
    if timeline.coordinate_frame != manifest.coordinate_frame:
        issues.append("mixed_coordinate_frame: timeline frame differs from the manifest")
    if timeline.model_manifest_ref != "manifest.json":
        issues.append("dangling_id: timeline model_manifest_ref must be manifest.json")
    for delta in deltas:
        if delta.edge is not None and delta.edge.coordinate_frame != frame:
            issues.append(f"mixed_coordinate_frame: edge {delta.edge.edge_id}")
    for sample in geometry:
        if sample.frame != frame:
            issues.append(f"mixed_coordinate_frame: geometry {sample.sample_id}")
    by_track: dict[str, list[float]] = {}
    for track in tracks:
        by_track.setdefault(track.track_id, []).append(track.t_sec)
    for track_id, times in by_track.items():
        if times != sorted(times):
            issues.append(f"schema: track {track_id} observations are not timestamp-monotonic")


def _check_references(
    timeline: SceneTimeline,
    tracks: list[TrackRecord],
    deltas: list[GraphDelta],
    proposals: list[Proposal],
    qa: list[QaRecord],
    geometry: list[GeometrySample],
    masks: list[MaskArtifact],
    root: Path,
    issues: list[str],
) -> None:
    observation_ids = _unique([row.observation_id for row in tracks], "observation_id", issues)
    track_ids = {row.track_id for row in tracks}
    delta_ids = _unique([row.delta_id for row in deltas], "delta_id", issues)
    proposal_ids = _unique([row.proposal_id for row in proposals], "proposal_id", issues)
    event_ids = _unique([row.event_id for row in timeline.events], "event_id", issues)
    qa_ids = _unique([row.qa_id for row in qa], "qa_id", issues)
    timeline_qa_ids = _unique([row.qa_id for row in timeline.qa_pairs], "timeline qa_id", issues)
    node_ids = _unique(
        [row.node.node_id for row in deltas if row.node is not None], "node_id", issues
    )
    edge_ids = _unique(
        [row.edge.edge_id for row in deltas if row.edge is not None], "edge_id", issues
    )
    sample_ids = _unique([row.sample_id for row in geometry], "sample_id", issues)
    known_entities = track_ids | node_ids
    evidence_ids = observation_ids | sample_ids

    if qa_ids != timeline_qa_ids:
        issues.append("dangling_id: qa.jsonl ids differ from timeline qa_pairs")
    logged = {row.qa_id: row for row in qa}
    for embedded in timeline.qa_pairs:
        record = logged.get(embedded.qa_id)
        if record is None:
            continue
        shared = embedded.model_dump()
        projected = {key: value for key, value in record.model_dump().items() if key in shared}
        if projected != shared:
            issues.append(f"schema: qa {record.qa_id} log differs from the timeline")

    for track in tracks:
        if track.identity_link is not None:
            _require(
                track.identity_link.linked_track_id,
                track_ids,
                f"identity_link on {track.observation_id}",
                issues,
            )
        _require(
            track.supersedes, observation_ids, f"track supersedes {track.observation_id}", issues
        )
    for delta in deltas:
        _require(delta.supersedes, delta_ids, f"delta supersedes {delta.delta_id}", issues)
        if delta.node is not None and delta.node.track_id is not None:
            _require(delta.node.track_id, track_ids, f"node {delta.node.node_id} track_id", issues)
        if delta.edge is not None:
            edge = delta.edge
            _require(edge.subject_id, node_ids, f"edge {edge.edge_id} subject", issues)
            _require(edge.object_id, node_ids, f"edge {edge.edge_id} object", issues)
            _require(edge.supersedes, edge_ids, f"edge supersedes {edge.edge_id}", issues)
            for evidence_id in edge.evidence_ids:
                _require(evidence_id, evidence_ids, f"edge {edge.edge_id} evidence", issues)
    for proposal in proposals:
        _require(
            proposal.supersedes, proposal_ids, f"proposal supersedes {proposal.proposal_id}", issues
        )
    for event in timeline.events:
        _require(event.supersedes, event_ids, f"event supersedes {event.event_id}", issues)
        for participant in event.participants:
            _require(participant, known_entities, f"event {event.event_id} participant", issues)
        for delta_id in event.state_delta_refs:
            _require(delta_id, delta_ids, f"event {event.event_id} state_delta", issues)
        if event.verification.vlm_claim_ref is not None:
            _require(
                event.verification.vlm_claim_ref,
                proposal_ids,
                f"event {event.event_id} vlm_claim_ref",
                issues,
            )
        for evidence in event.evidence:
            for track_id in evidence.track_ids:
                _require(track_id, track_ids, f"event {event.event_id} evidence track", issues)
            for ref in (evidence.mask_ref, evidence.geometry_ref):
                if ref is not None and not (root / ref).is_file():
                    issues.append(f"dangling_id: event {event.event_id} missing {ref}")
    for pair in timeline.qa_pairs:
        for event_id in pair.evidence_event_ids:
            _require(event_id, event_ids, f"qa {pair.qa_id} evidence", issues)
    for mask in masks:
        _require(mask.track_id, track_ids, f"mask {mask.track_id}", issues)
    _supersession_cycles([(row.observation_id, row.supersedes) for row in tracks], "tracks", issues)
    _supersession_cycles([(row.delta_id, row.supersedes) for row in deltas], "deltas", issues)
    _supersession_cycles(
        [(row.event_id, row.supersedes) for row in timeline.events], "events", issues
    )


def _check_geometry(
    manifest: AnalysisManifest,
    deltas: list[GraphDelta],
    geometry: list[GeometrySample],
    issues: list[str],
) -> None:
    grouped = _geometry_index(geometry)
    for subject_id, samples in grouped.items():
        series = [(sample.t_sec, sample.center_m) for sample in samples]
        if not velocity_feasible(series):
            issues.append(f"schema: infeasible velocity for {subject_id}")
    metric = manifest.coordinate_frame.metric_scale == "metric"
    for delta in deltas:
        edge = delta.edge
        if edge is None or edge.verification_status != "accepted":
            continue
        if not metric and edge.predicate in _METRIC_PREDICATES:
            issues.append(
                f"schema: metric predicate {edge.predicate} on edge {edge.edge_id} "
                "requires metric_scale metric"
            )
        subject = _sample_in_interval(
            grouped.get(edge.subject_id, []), edge.start_sec, edge.end_sec
        )
        obj = _sample_in_interval(grouped.get(edge.object_id, []), edge.start_sec, edge.end_sec)
        if subject is None or obj is None:
            issues.append(
                f"dangling_id: accepted edge {edge.edge_id} has no geometry in its interval"
            )
            continue
        band = None
        if edge.distance_band is not None:
            band = (edge.distance_band.min_m, edge.distance_band.max_m)
        holds = relation_holds(
            edge.predicate,
            subject.center_m,
            subject.extent_m,
            obj.center_m,
            obj.extent_m,
            distance_band=band,
            subject_depth_m=subject.depth_m,
            object_depth_m=obj.depth_m,
            subject_motion_m=_motion_m(
                grouped.get(edge.subject_id, []), edge.start_sec, edge.end_sec
            ),
        )
        if not holds:
            issues.append(
                f"schema: accepted edge {edge.edge_id} predicate {edge.predicate} "
                "contradicts geometry"
            )


def _check_gaps(manifest: AnalysisManifest, gaps: list[GapRecord], issues: list[str]) -> None:
    skipped = sum(stage.skipped_frames for stage in manifest.stages)
    covered = sum(gap.skipped_frames for gap in gaps)
    if skipped > 0 and covered < skipped:
        issues.append(
            "schema: skipped frames require a gap record covering each discarded interval"
        )
    if skipped == 0 and gaps:
        issues.append("schema: gap records require a stage with skipped_frames > 0")
