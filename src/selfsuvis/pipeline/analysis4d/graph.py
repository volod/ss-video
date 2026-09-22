"""Append-only temporal scene graph.

Nodes and edges are graph deltas. Replaying those deltas is the current
graph. Relation intervals are closed only after the observations that
support them, in time order.
"""

from dataclasses import dataclass, field

from selfsuvis.pipeline.analysis4d.geometry import center_error_m, relation_holds
from selfsuvis.pipeline.analysis4d.schemas import (
    DETERMINISTIC_PREDICATES,
    SCHEMA_DELTA,
    DistanceBand,
    EvidenceRef,
    GeometrySample,
    GraphDelta,
    GraphEdge,
    GraphNode,
    Location3D,
    TimelineEvent,
    TrackRecord,
    Verification,
)

HOLD_TAIL_SEC = 1.0
EVENT_SEC = 1.0
_METRIC = frozenset({"distance_band", "supports", "contacts"})
_SYMMETRIC = frozenset({"intersects", "contacts", "distance_band"})
_BANDS = ((0.0, 2.0), (2.0, 8.0), (8.0, 32.0))
_ACTIVE = frozenset({"tentative", "confirmed", "occluded"})
_APPROACH_M = 2.0


def event_token(value: str) -> str:
    """Return a contract id token. Separators in ``value`` become hyphens."""
    cleaned = "".join(char if char.isalnum() else "-" for char in value.lower())
    token = "-".join(part for part in cleaned.split("-") if part)
    return (token or "label")[:64]


@dataclass(frozen=True)
class RegionBox:
    """Static region the reducer can write when the mission has no region sample yet."""

    node_id: str
    label: str
    center_m: list[float]
    extent_m: list[float]


@dataclass
class GraphView:
    """Materialized graph. Superseded deltas stay in the log and leave this view."""

    nodes: list[GraphNode] = field(default_factory=list)
    superseded_nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    superseded_delta_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReducedGraph:
    """Deltas and the timeline events those deltas support."""

    deltas: list[GraphDelta]
    events: list[TimelineEvent]


def materialize(deltas: list[GraphDelta]) -> GraphView:
    """Replay deltas in ``(t_sec, delta_id)`` order.

    Args:
        deltas: Append-only graph log. Input order does not change the view.

    Returns:
        Current nodes and edges, plus nodes whose add delta was superseded.
    """
    ordered = sorted(deltas, key=lambda row: (row.t_sec, row.delta_id))
    superseded: set[str] = set()
    for delta in ordered:
        if delta.supersedes:
            superseded.add(delta.supersedes)
    nodes: list[GraphNode] = []
    old_nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for delta in ordered:
        if delta.node is not None:
            bucket = old_nodes if delta.delta_id in superseded else nodes
            bucket.append(delta.node)
        if delta.edge is not None and delta.delta_id not in superseded and delta.op != "end":
            edges.append(delta.edge)
    return GraphView(
        nodes=nodes,
        superseded_nodes=old_nodes,
        edges=edges,
        superseded_delta_ids=sorted(superseded),
    )


def reduce_graph(
    mission_id: str,
    tracks: list[TrackRecord],
    samples: list[GeometrySample],
    *,
    frame_name: str,
    metric: bool,
    region_ids: set[str] | None = None,
) -> ReducedGraph:
    """Merge tracks and geometry into nodes, relation intervals, and events.

    Args:
        mission_id: Mission stored on every delta.
        tracks: 2D observations. Identity links supersede the linked node.
        samples: 3D samples. ``subject_id`` is the node id.
        frame_name: Coordinate frame stored on edges. It must match the manifest.
        metric: When false, metric predicates are omitted.
        region_ids: Node ids that are regions. Other non-track subjects are regions too.

    Returns:
        Deltas and accepted events. Open-vocabulary claims are not created here.
    """
    by_subject = _index_samples(samples)
    track_ids = {row.track_id for row in tracks}
    regions = set(region_ids or ())
    regions.update(subject for subject in by_subject if subject not in track_ids)
    labels = _labels(tracks)
    first_t = _first_times(tracks, by_subject)
    links = _links(tracks)
    deltas: list[GraphDelta] = []
    node_delta: dict[str, str] = {}
    for node_id in sorted(first_t, key=lambda item: (first_t[item], item)):
        kind = "region" if node_id in regions else "track"
        delta_id = f"delta-node-{node_id}"
        linked = links.get(node_id)
        supersedes = node_delta.get(linked) if linked else None
        op = "supersede" if supersedes else "add"
        deltas.append(
            GraphDelta(
                schema_version=SCHEMA_DELTA,
                mission_id=mission_id,
                delta_id=delta_id,
                op=op,  # type: ignore[arg-type]
                t_sec=first_t[node_id],
                node=GraphNode(
                    node_id=node_id,
                    kind=kind,  # type: ignore[arg-type]
                    label=labels.get(node_id, node_id),
                    track_id=node_id if node_id in track_ids else None,
                ),
                supersedes=supersedes,
            )
        )
        node_delta[node_id] = delta_id
    edges = _relation_edges(
        mission_id,
        by_subject,
        frame_name=frame_name,
        metric=metric,
        track_ids=track_ids,
        region_ids=regions,
        active=_active_at(tracks),
    )
    deltas.extend(edges)
    events = _events(
        mission_id,
        edges,
        tracks,
        by_subject,
        frame_name=frame_name,
        metric=metric,
        region_ids=regions,
        node_delta=node_delta,
    )
    return ReducedGraph(deltas=deltas, events=events)


def serialize_context(
    tracks: list[TrackRecord],
    samples: list[GeometrySample],
    view: GraphView,
) -> dict:
    """Serialize regions, trajectories, and edges for a proposal provider.

    Args:
        tracks: Track labels.
        samples: Geometry samples, including regions.
        view: Materialized graph.

    Returns:
        A JSON-ready object. Keys are sorted by the caller when it is encoded.
    """
    labels = _labels(tracks)
    trajectories: dict[str, list[dict]] = {}
    regions: list[dict] = []
    track_nodes = {node.node_id for node in view.nodes if node.kind == "track"}
    for sample in sorted(samples, key=lambda row: (row.subject_id, row.t_sec, row.sample_id)):
        point = {
            "t_sec": sample.t_sec,
            "center_m": list(sample.center_m),
            "extent_m": list(sample.extent_m),
        }
        if sample.subject_id in labels or sample.subject_id in track_nodes:
            trajectories.setdefault(sample.subject_id, []).append(point)
        else:
            regions.append({"node_id": sample.subject_id, **point})
    return {
        "regions": regions,
        "trajectories": [
            {"track_id": track_id, "label": labels.get(track_id, track_id), "samples": points}
            for track_id, points in sorted(trajectories.items())
        ],
        "edges": [
            {
                "subject_id": edge.subject_id,
                "predicate": edge.predicate,
                "object_id": edge.object_id,
                "start_sec": edge.start_sec,
                "end_sec": edge.end_sec,
            }
            for edge in view.edges
        ],
    }


def _relation_edges(
    mission_id: str,
    by_subject: dict[str, list[GeometrySample]],
    *,
    frame_name: str,
    metric: bool,
    track_ids: set[str],
    region_ids: set[str],
    active: dict[float, set[str]],
) -> list[GraphDelta]:
    subjects = sorted(by_subject)
    found: list[tuple[float, float, str, str, str, DistanceBand | None, str]] = []
    for left in subjects:
        for right in subjects:
            if left == right:
                continue
            flags = []
            for stamp, sample_a, sample_b in _co_times(by_subject[left], by_subject[right]):
                flags.append(
                    (
                        stamp,
                        sample_a.sample_id,
                        _predicates_at(
                            left,
                            right,
                            sample_a,
                            sample_b,
                            metric=metric,
                            track_ids=track_ids,
                            region_ids=region_ids,
                            active=active.get(round(stamp, 6), set()),
                        ),
                    )
                )
            found.extend(_collapse(left, right, flags))
    found.extend(_motion_edges(by_subject))
    deltas: list[GraphDelta] = []
    for start, end, subject_id, predicate, object_id, band, evidence_id in sorted(found):
        if predicate not in DETERMINISTIC_PREDICATES or not evidence_id:
            continue
        millis = int(round(start * 1000))
        edge_id = f"edge-{predicate}-{subject_id}-{object_id}-{millis:07d}"
        deltas.append(
            GraphDelta(
                schema_version=SCHEMA_DELTA,
                mission_id=mission_id,
                delta_id=f"delta-{edge_id}",
                op="add",
                t_sec=start,
                edge=GraphEdge(
                    edge_id=edge_id,
                    subject_id=subject_id,
                    predicate=predicate,
                    object_id=object_id,
                    start_sec=start,
                    end_sec=end,
                    coordinate_frame=frame_name,
                    confidence=0.95,
                    source="deterministic",
                    evidence_ids=[evidence_id],
                    verification_status="accepted",
                    distance_band=band if predicate == "distance_band" else None,
                ),
            )
        )
    return deltas


def _predicates_at(
    left: str,
    right: str,
    sample_a: GeometrySample,
    sample_b: GeometrySample,
    *,
    metric: bool,
    track_ids: set[str],
    region_ids: set[str],
    active: set[str],
) -> list[tuple[str, DistanceBand | None]]:
    allow_metric = (
        metric and sample_a.metric_scale == "metric" and sample_b.metric_scale == "metric"
    )
    distance = center_error_m(sample_a.center_m, sample_b.center_m)
    band = _band_for(distance) if allow_metric else None
    found: list[tuple[str, DistanceBand | None]] = []
    for predicate in (
        "contains",
        "intersects",
        "left_of",
        "right_of",
        "above",
        "below",
        "supports",
        "contacts",
        "occludes",
        "distance_band",
    ):
        if predicate in _SYMMETRIC and left >= right:
            continue
        if predicate in _METRIC and not allow_metric:
            continue
        payload = band if predicate == "distance_band" else None
        if payload is None and predicate == "distance_band":
            continue
        holds = relation_holds(
            predicate,
            sample_a.center_m,
            sample_a.extent_m,
            sample_b.center_m,
            sample_b.extent_m,
            distance_band=None if payload is None else (payload.min_m, payload.max_m),
            subject_depth_m=sample_a.depth_m,
            object_depth_m=sample_b.depth_m,
        )
        if holds:
            found.append((predicate, payload))
    if left in track_ids and right in region_ids and left in active:
        found.append(("visible", None))
    return found


def _collapse(
    left: str,
    right: str,
    flags: list[tuple[float, str, list[tuple[str, DistanceBand | None]]]],
) -> list[tuple[float, float, str, str, str, DistanceBand | None, str]]:
    """Merge consecutive timestamps that share one predicate into one interval."""
    keys: set[tuple[str, str]] = set()
    for _stamp, _sample_id, preds in flags:
        for predicate, band in preds:
            keys.add((predicate, _band_token(band)))
    rows = []
    times = [stamp for stamp, _sample_id, _preds in flags]
    for predicate, token in sorted(keys):
        holds: list[bool] = []
        evidence_by_t: dict[float, str] = {}
        band_by_t: dict[float, DistanceBand | None] = {}
        for stamp, sample_id, preds in flags:
            ok, matched_band = _match_predicate(preds, predicate, token)
            holds.append(ok)
            if ok:
                evidence_by_t[stamp] = sample_id
                band_by_t[stamp] = matched_band
        for start, end in _runs(times, holds):
            inside = [stamp for stamp in times if start <= stamp < end and stamp in evidence_by_t]
            if not inside:
                continue
            last = inside[-1]
            rows.append((start, end, left, predicate, right, band_by_t[last], evidence_by_t[last]))
    return rows


def _match_predicate(
    preds: list[tuple[str, DistanceBand | None]],
    predicate: str,
    token: str,
) -> tuple[bool, DistanceBand | None]:
    """Return whether ``preds`` contains this predicate and its distance band."""
    for candidate, band in preds:
        if candidate == predicate and _band_token(band) == token:
            return True, band
    return False, None


def _band_token(band: DistanceBand | None) -> str:
    if band is None:
        return ""
    return f"{band.min_m:.3f}-{band.max_m:.3f}"


def _runs(times: list[float], holds: list[bool]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for stamp, ok in zip(times, holds, strict=True):
        if ok and start is None:
            start = stamp
        if ok:
            continue
        if start is not None:
            intervals.append((start, stamp))
            start = None
    if start is not None and times:
        intervals.append((start, times[-1] + HOLD_TAIL_SEC))
    return intervals


def _motion_edges(
    by_subject: dict[str, list[GeometrySample]],
) -> list[tuple[float, float, str, str, str, DistanceBand | None, str]]:
    rows = []
    subjects = sorted(by_subject)
    for subject in subjects:
        samples = by_subject[subject]
        if len(samples) < 2:
            continue
        start = samples[0].t_sec
        end = samples[-1].t_sec + HOLD_TAIL_SEC
        moved = center_error_m(samples[0].center_m, samples[-1].center_m)
        if moved <= 0.05:
            continue
        for other in subjects:
            if other == subject:
                continue
            if not any(start <= sample.t_sec < end for sample in by_subject[other]):
                continue
            rows.append(
                (
                    start,
                    end,
                    subject,
                    "relative_motion",
                    other,
                    None,
                    samples[-1].sample_id,
                )
            )
            break
    return rows


def _events(
    mission_id: str,
    edges: list[GraphDelta],
    tracks: list[TrackRecord],
    by_subject: dict[str, list[GeometrySample]],
    *,
    frame_name: str,
    metric: bool,
    region_ids: set[str],
    node_delta: dict[str, str],
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for delta in edges:
        edge = delta.edge
        if edge is None:
            continue
        if edge.predicate == "contains" and edge.subject_id in region_ids:
            events.append(
                _event(
                    mission_id,
                    "entered_region",
                    f"{edge.object_id} entered {edge.subject_id}",
                    edge,
                    [edge.object_id, edge.subject_id],
                    by_subject,
                    frame_name=frame_name,
                    metric=metric,
                    rules=["contains"],
                    region_ids=region_ids,
                )
            )
            later = [
                sample.t_sec
                for sample in by_subject.get(edge.object_id, [])
                if sample.t_sec >= edge.end_sec
            ]
            if later:
                events.append(
                    _event(
                        mission_id,
                        "left_region",
                        f"{edge.object_id} left {edge.subject_id}",
                        edge,
                        [edge.object_id, edge.subject_id],
                        by_subject,
                        frame_name=frame_name,
                        metric=metric,
                        rules=["contains"],
                        region_ids=region_ids,
                        at_sec=later[0],
                    )
                )
        if (
            edge.predicate == "distance_band"
            and edge.distance_band is not None
            and edge.distance_band.max_m <= _APPROACH_M
        ):
            events.append(
                _event(
                    mission_id,
                    "approached",
                    f"{edge.subject_id} approached {edge.object_id}",
                    edge,
                    [edge.subject_id, edge.object_id],
                    by_subject,
                    frame_name=frame_name,
                    metric=metric,
                    rules=["distance_band"],
                    region_ids=region_ids,
                )
            )
        if edge.predicate == "supports":
            events.append(
                _event(
                    mission_id,
                    "put_down",
                    f"{edge.object_id} put down on {edge.subject_id}",
                    edge,
                    [edge.object_id, edge.subject_id],
                    by_subject,
                    frame_name=frame_name,
                    metric=metric,
                    rules=["supports"],
                    region_ids=region_ids,
                )
            )
    events.extend(
        _picked_up(
            mission_id,
            edges,
            by_subject,
            frame_name=frame_name,
            metric=metric,
            region_ids=region_ids,
        )
    )
    events.extend(
        _counts(
            mission_id,
            tracks,
            by_subject,
            frame_name=frame_name,
            metric=metric,
            node_delta=node_delta,
        )
    )
    return sorted(events, key=lambda event: (event.start_sec, event.type, event.event_id))


def _picked_up(
    mission_id: str,
    edges: list[GraphDelta],
    by_subject: dict[str, list[GeometrySample]],
    *,
    frame_name: str,
    metric: bool,
    region_ids: set[str],
) -> list[TimelineEvent]:
    events = []
    for delta in edges:
        edge = delta.edge
        if edge is None or edge.predicate != "supports":
            continue
        samples = by_subject.get(edge.object_id, [])
        later = [sample for sample in samples if sample.t_sec >= edge.end_sec - 1e-9]
        if len(samples) < 2 or not later:
            continue
        moved = center_error_m(samples[0].center_m, later[0].center_m)
        if moved <= 0.05:
            continue
        events.append(
            _event(
                mission_id,
                "picked_up",
                f"{edge.object_id} picked up from {edge.subject_id}",
                edge,
                [edge.object_id, edge.subject_id],
                by_subject,
                frame_name=frame_name,
                metric=metric,
                rules=["relative_motion", "supports"],
                region_ids=region_ids,
                at_sec=later[0].t_sec,
            )
        )
    return events


def _counts(
    mission_id: str,
    tracks: list[TrackRecord],
    by_subject: dict[str, list[GeometrySample]],
    *,
    frame_name: str,
    metric: bool,
    node_delta: dict[str, str],
) -> list[TimelineEvent]:
    by_time: dict[float, dict[str, set[str]]] = {}
    for record in sorted(tracks, key=lambda row: (row.t_sec, row.track_id)):
        if record.state not in _ACTIVE:
            continue
        bucket = by_time.setdefault(round(record.t_sec, 6), {})
        bucket.setdefault(record.label_normalized, set()).add(record.track_id)
    previous: dict[str, set[str]] = {}
    events: list[TimelineEvent] = []
    for stamp, groups in sorted(by_time.items()):
        labels = set(previous) | set(groups)
        for label in sorted(labels):
            current = groups.get(label, set())
            if current == previous.get(label, set()):
                continue
            if not current:
                previous[label] = current
                continue
            participant = sorted(current)[0]
            samples = by_subject.get(participant, [])
            sample = next((item for item in samples if abs(item.t_sec - stamp) < 1e-6), None)
            if sample is None and samples:
                sample = samples[0]
            if sample is None:
                previous[label] = current
                continue
            end = stamp + EVENT_SEC
            if end <= stamp:
                previous[label] = current
                continue
            delta_ids = [
                node_delta[track_id] for track_id in sorted(current) if track_id in node_delta
            ]
            events.append(
                TimelineEvent(
                    event_id=f"evt-count-{event_token(label)}-{int(round(stamp * 1000)):07d}",
                    type="count_changed",
                    summary=f"{label} count changed to {len(current)}",
                    start_sec=stamp,
                    end_sec=end,
                    participants=sorted(current),
                    state_delta_refs=delta_ids,
                    location=_location(sample, frame_name) if metric else None,
                    confidence=0.9,
                    verification=Verification(status="accepted", rules=["track_count"], reasons=[]),
                    evidence=[_evidence(mission_id, participant, sample)],
                )
            )
            previous[label] = set(current)
    return events


def _event(
    mission_id: str,
    event_type: str,
    summary: str,
    edge: GraphEdge,
    participants: list[str],
    by_subject: dict[str, list[GeometrySample]],
    *,
    frame_name: str,
    metric: bool,
    rules: list[str],
    region_ids: set[str],
    at_sec: float | None = None,
) -> TimelineEvent:
    stamp = edge.start_sec if at_sec is None else at_sec
    track_id = next(item for item in participants if item not in region_ids)
    samples = by_subject.get(track_id, [])
    sample = next((item for item in samples if abs(item.t_sec - stamp) < 1e-6), None)
    if sample is None and samples:
        sample = min(samples, key=lambda item: abs(item.t_sec - stamp))
    if sample is None:
        raise ValueError(f"event {event_type} for {track_id} has no geometry")
    start = stamp
    end = start + EVENT_SEC
    millis = int(round(start * 1000))
    other = next(item for item in participants if item != track_id)
    return TimelineEvent(
        event_id=f"evt-{event_type}-{track_id}-{other}-{millis:07d}",
        type=event_type,
        summary=summary,
        start_sec=start,
        end_sec=end,
        participants=participants,
        state_delta_refs=[f"delta-{edge.edge_id}"],
        location=_location(sample, frame_name) if metric else None,
        confidence=0.9,
        verification=Verification(status="accepted", rules=rules, reasons=[]),
        evidence=[_evidence(mission_id, track_id, sample)],
    )


def _evidence(mission_id: str, track_id: str, sample: GeometrySample) -> EvidenceRef:
    millis = int(round(sample.t_sec * 1000))
    return EvidenceRef(
        frame_id=f"{mission_id}:{track_id}:{millis:07d}",
        t_sec=sample.t_sec,
        track_ids=[track_id],
        geometry_ref=f"geometry/{sample.subject_id}/{millis:07d}.json",
    )


def _location(sample: GeometrySample, frame_name: str) -> Location3D:
    covariance = sample.covariance_diag or [0.01, 0.01, 0.01]
    return Location3D(
        frame=frame_name, center_m=list(sample.center_m), covariance_diag=list(covariance)
    )


def _index_samples(samples: list[GeometrySample]) -> dict[str, list[GeometrySample]]:
    grouped: dict[str, list[GeometrySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.subject_id, []).append(sample)
    for rows in grouped.values():
        rows.sort(key=lambda sample: sample.t_sec)
    return grouped


def _co_times(
    left: list[GeometrySample], right: list[GeometrySample]
) -> list[tuple[float, GeometrySample, GeometrySample]]:
    right_by_t = {round(sample.t_sec, 6): sample for sample in right}
    pairs = []
    for sample in left:
        other = right_by_t.get(round(sample.t_sec, 6))
        if other is not None:
            pairs.append((sample.t_sec, sample, other))
    return pairs


def _labels(tracks: list[TrackRecord]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for record in sorted(tracks, key=lambda row: (row.t_sec, row.observation_id)):
        labels[record.track_id] = record.label_normalized
    return labels


def _first_times(
    tracks: list[TrackRecord], by_subject: dict[str, list[GeometrySample]]
) -> dict[str, float]:
    first: dict[str, float] = {}
    for record in tracks:
        first[record.track_id] = min(record.t_sec, first.get(record.track_id, record.t_sec))
    for subject, samples in by_subject.items():
        if samples and subject not in first:
            first[subject] = samples[0].t_sec
    return first


def _links(tracks: list[TrackRecord]) -> dict[str, str]:
    linked: dict[str, str] = {}
    for record in tracks:
        if record.identity_link is not None:
            linked[record.track_id] = record.identity_link.linked_track_id
    return linked


def _active_at(tracks: list[TrackRecord]) -> dict[float, set[str]]:
    active: dict[float, set[str]] = {}
    for record in tracks:
        if record.state not in _ACTIVE:
            continue
        active.setdefault(round(record.t_sec, 6), set()).add(record.track_id)
    return active


def _band_for(distance: float) -> DistanceBand | None:
    for low, high in _BANDS:
        if low <= distance <= high:
            return DistanceBand(min_m=low, max_m=high)
    return None


def _evidence_id(subject_id: str, stamp: float) -> str:
    del subject_id, stamp
    return ""
