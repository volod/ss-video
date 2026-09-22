"""Deterministic strict verifier for timeline claims.

Geometry is recomputed from stored samples. A measurable predicate whose
calibration gate passes is accepted or rejected from that measurement. Model
confidence cannot override it. Semantic claims that geometry does not settle
go to a reviewer with measurements and counter-evidence. Reviewer timeout,
refusal, and malformed output leave those claims uncertain and do not drop
deterministic results.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from selfsuvis.pipeline.analysis4d.geometry import relation_holds, velocity_feasible
from selfsuvis.pipeline.analysis4d.graph import event_token
from selfsuvis.pipeline.analysis4d.review import (
    ReviewDecision,
    ReviewError,
    ReviewPacket,
    ReviewProvider,
    UnavailableReview,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    DETERMINISTIC_PREDICATES,
    SCHEMA_PROPOSAL,
    EvidenceRef,
    GeometrySample,
    GraphDelta,
    GraphEdge,
    Location3D,
    Proposal,
    TimelineEvent,
    TrackRecord,
    Verification,
    VerificationStatus,
)

_RESIDUAL_MAX_M = 0.5
_COVARIANCE_MAX = 1.0
_METRIC = frozenset({"distance_band", "supports", "contacts"})
_OPPOSITE = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above"}
_ACTIVE = frozenset({"tentative", "confirmed", "occluded"})
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_BAND = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meters?)?\s*(?:to|-)\s*(\d+(?:\.\d+)?)")


@dataclass
class Resolution:
    """How one proposal was resolved. ``needs_review`` claims are not settled yet."""

    proposal_id: str
    status: VerificationStatus
    rules: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    needs_review: bool = False
    subject_id: str | None = None
    object_id: str | None = None
    predicate: str | None = None


@dataclass
class VerifierOutput:
    """Resolutions plus the degradation recorded when review fails closed."""

    resolutions: list[Resolution]
    degradations: list[str] = field(default_factory=list)
    review_failure: str | None = None


def verify_proposals(
    proposals: list[Proposal],
    *,
    tracks: list[TrackRecord],
    deltas: list[GraphDelta],
    samples: list[GeometrySample],
    metric: bool,
    frame: str,
    reviewer: ReviewProvider | None = None,
    events: list[TimelineEvent] | None = None,
) -> VerifierOutput:
    """Resolve proposals. Geometry runs before any reviewer call.

    Args:
        proposals: Audit log, including claims a model already marked accepted.
        tracks: Track observations used for counts and evidence.
        deltas: Graph log used for node ids and opposite edges.
        samples: Stored geometry. Predicates are recomputed from these.
        metric: True when the manifest scale is ``metric``.
        frame: Mission coordinate frame name.
        reviewer: Semantic reviewer. The default is unavailable.
        events: Accepted events already on the timeline. An action claim can
            be accepted only when one of these intervals covers it.

    Returns:
        Resolutions. A failed reviewer leaves semantic claims uncertain.
    """
    chosen = reviewer if reviewer is not None else UnavailableReview()
    grouped = _index(samples)
    nodes = _node_ids(deltas)
    settled: list[Resolution] = []
    pending: list[tuple[Proposal, Resolution]] = []
    for proposal in proposals:
        resolution = _resolve_one(
            proposal,
            tracks=tracks,
            deltas=deltas,
            grouped=grouped,
            nodes=nodes,
            metric=metric,
            frame=frame,
        )
        if resolution.needs_review:
            pending.append((proposal, resolution))
        else:
            settled.append(resolution)
    failure, decisions = _review(chosen, pending, grouped)
    by_id = {item.proposal_id: item for item in decisions}
    aliases = _participant_aliases(deltas)
    for proposal, resolution in pending:
        settled.append(
            _apply_decision(
                proposal,
                resolution,
                by_id.get(proposal.proposal_id),
                failure=failure,
                tracks=tracks,
                samples=samples,
                events=events or [],
                aliases=aliases,
            )
        )
    order = {proposal.proposal_id: index for index, proposal in enumerate(proposals)}
    settled.sort(key=lambda item: order.get(item.proposal_id, 0))
    flags = ["provider_unavailable"] if getattr(chosen, "failed", False) or failure else []
    return VerifierOutput(resolutions=settled, degradations=flags, review_failure=failure)


def recheck_events(
    events: list[TimelineEvent],
    *,
    deltas: list[GraphDelta],
    samples: list[GeometrySample],
    tracks: list[TrackRecord],
    metric: bool,
    root: Path,
) -> list[TimelineEvent]:
    """Recompute evidence for existing events. Accepted events keep their times.

    Args:
        events: Timeline events written by an earlier stage.
        deltas: Graph log. Edge deltas are recomputed.
        samples: Stored geometry.
        tracks: Used when the event rule is ``track_count``.
        metric: Manifest metric gate.
        root: Artifact directory. Evidence paths must exist under it.

    Returns:
        Events with status ``accepted``, ``rejected``, or ``uncertain``.
    """
    grouped = _index(samples)
    by_delta = {delta.delta_id: delta for delta in deltas}
    checked: list[TimelineEvent] = []
    for event in events:
        status, rules, reasons = _event_status(
            event,
            by_delta=by_delta,
            grouped=grouped,
            tracks=tracks,
            metric=metric,
            root=root,
        )
        verification = event.verification.model_copy(
            update={
                "status": status,
                "rules": rules or list(event.verification.rules),
                "reasons": reasons,
            }
        )
        checked.append(event.model_copy(update={"verification": verification}))
    return checked


def count_events(
    resolutions: list[Resolution],
    *,
    mission_id: str,
    proposals: list[Proposal],
    tracks: list[TrackRecord],
    deltas: list[GraphDelta],
    samples: list[GeometrySample],
    events: list[TimelineEvent],
    frame: str,
    metric: bool,
    root: Path,
) -> list[TimelineEvent]:
    """Add a count event when a count claim is accepted and no event covers it.

    Args:
        resolutions: Verifier output.
        mission_id: Mission id stored on evidence frame ids.
        proposals: Original proposals, used to read the count text.
        tracks: Observations that were counted.
        deltas: Node deltas cited by the new event.
        samples: Geometry used as evidence.
        events: Events already on the timeline.
        frame: Coordinate frame for a metric location.
        metric: When false, the event has no metric location.
        root: Artifact directory used to find a geometry file.

    Returns:
        New accepted count events. Existing count events are not duplicated.
    """
    by_id = {proposal.proposal_id: proposal for proposal in proposals}
    created: list[TimelineEvent] = []
    for resolution in resolutions:
        if resolution.status != "accepted" or "track_count" not in resolution.rules:
            continue
        proposal = by_id.get(resolution.proposal_id)
        if proposal is None:
            continue
        parsed = _parse_count(proposal.text)
        if parsed is None:
            continue
        number, label = parsed
        counted = _tracks_for_label(tracks, label, proposal.start_sec or 0.0)
        if len(counted) != number or not counted:
            continue
        participants = sorted(counted)
        if any(
            event.type == "count_changed" and sorted(event.participants) == participants
            for event in events
        ):
            continue
        sample = _sample_for_track(participants[0], deltas, samples)
        if sample is None:
            continue
        ref = _geometry_ref(root, sample)
        if ref is None:
            continue
        stamp = proposal.start_sec or sample.t_sec
        millis = int(round(stamp * 1000))
        event_id = f"evt-count-{event_token(label)}-{millis:07d}"
        if any(event.event_id == event_id for event in events):
            continue
        delta_ids = _node_deltas(deltas, participants)
        created.append(
            TimelineEvent(
                event_id=event_id,
                type="count_changed",
                summary=f"{label} count changed to {number}",
                start_sec=stamp,
                end_sec=stamp + 1.0,
                participants=participants,
                state_delta_refs=delta_ids,
                location=_location(sample, frame) if metric else None,
                confidence=0.9,
                verification=Verification(status="accepted", rules=["track_count"], reasons=[]),
                evidence=[
                    EvidenceRef(
                        frame_id=f"{mission_id}:{participants[0]}:{millis:07d}",
                        t_sec=sample.t_sec,
                        track_ids=[participants[0]],
                        geometry_ref=ref,
                    )
                ],
            )
        )
    return created


def audit_proposal(mission_id: str, proposal: Proposal, resolution: Resolution) -> Proposal | None:
    """Return a superseding audit row when the resolution changes the stored status.

    The original row stays in the log. The new row links to it with ``supersedes``.
    """
    if resolution.status == proposal.verification_status and resolution.reasons == list(
        proposal.reasons
    ):
        return None
    return Proposal(
        schema_version=SCHEMA_PROPOSAL,
        mission_id=mission_id,
        proposal_id=f"audit-{proposal.proposal_id}",
        claim_kind=proposal.claim_kind,
        text=proposal.text,
        subject_id=resolution.subject_id or proposal.subject_id,
        predicate=resolution.predicate or proposal.predicate,
        object_id=resolution.object_id or proposal.object_id,
        start_sec=proposal.start_sec,
        end_sec=proposal.end_sec,
        verification_status=resolution.status,
        supersedes=proposal.proposal_id,
        reasons=list(resolution.reasons),
    )


def effective_status(proposals: list[Proposal], proposal_id: str) -> str | None:
    """Return the latest status in a supersession chain.

    Args:
        proposals: Audit log in file order.
        proposal_id: The claim whose effective status is required.

    Returns:
        The status of the latest row that supersedes ``proposal_id``, or None
        when the id is missing.
    """
    by_id = {proposal.proposal_id: proposal for proposal in proposals}
    if proposal_id not in by_id:
        return None
    children: dict[str, list[Proposal]] = {}
    for proposal in proposals:
        if proposal.supersedes:
            children.setdefault(proposal.supersedes, []).append(proposal)
    cursor = proposal_id
    seen = {cursor}
    while children.get(cursor):
        cursor = children[cursor][-1].proposal_id
        if cursor in seen:
            break
        seen.add(cursor)
    return by_id[cursor].verification_status


def publishable_events(events: list[TimelineEvent]) -> list[TimelineEvent]:
    """Return accepted events that cite evidence. Rejected and uncertain stay out."""
    return [event for event in events if event.verification.status == "accepted" and event.evidence]


def accepted_edges(deltas: list[GraphDelta]) -> list[GraphEdge]:
    """Return edges whose stored status is accepted."""
    return [
        delta.edge
        for delta in deltas
        if delta.edge is not None and delta.edge.verification_status == "accepted"
    ]


def _resolve_one(
    proposal: Proposal,
    *,
    tracks: list[TrackRecord],
    deltas: list[GraphDelta],
    grouped: dict[str, list[GeometrySample]],
    nodes: set[str],
    metric: bool,
    frame: str,
) -> Resolution:
    subject, obj = _bind(proposal, nodes)
    predicate = proposal.predicate
    base = Resolution(
        proposal_id=proposal.proposal_id,
        status="uncertain",
        subject_id=subject,
        object_id=obj,
        predicate=predicate,
    )
    if _is_identity(proposal):
        if _identity_link(tracks, subject):
            return _done(base, "accepted", ["identity_link"], [])
        return _done(base, "uncertain", [], ["vlm_not_sole_evidence"])
    parsed = _parse_count(proposal.text)
    if parsed is not None and predicate not in DETERMINISTIC_PREDICATES:
        number, label = parsed
        at = proposal.start_sec if proposal.start_sec is not None else 0.0
        actual = len(_tracks_for_label(tracks, label, at))
        if actual == number and number > 0:
            return _done(base, "accepted", ["track_count"], [])
        return _done(base, "rejected", ["track_count"], ["count_mismatch"])
    if predicate in DETERMINISTIC_PREDICATES:
        return _geometry(
            proposal,
            base,
            grouped=grouped,
            nodes=nodes,
            deltas=deltas,
            tracks=tracks,
            metric=metric,
            frame=frame,
        )
    if proposal.claim_kind == "event":
        return _done(base, "uncertain", [], ["vlm_not_sole_evidence"])
    for endpoint in (subject, obj):
        if endpoint is not None and endpoint not in nodes and endpoint not in _track_ids(tracks):
            return _done(base, "rejected", [], ["dangling_id"])
    base.needs_review = True
    return base


def _geometry(
    proposal: Proposal,
    base: Resolution,
    *,
    grouped: dict[str, list[GeometrySample]],
    nodes: set[str],
    deltas: list[GraphDelta],
    tracks: list[TrackRecord],
    metric: bool,
    frame: str,
) -> Resolution:
    del frame
    predicate = proposal.predicate or ""
    subject = base.subject_id
    obj = base.object_id
    known = nodes | _track_ids(tracks) | set(grouped)
    if not subject or not obj or subject not in known or obj not in known:
        return _done(base, "rejected", [predicate], ["dangling_id"])
    left = grouped.get(subject, [])
    right = grouped.get(obj, [])
    if proposal.start_sec is not None and proposal.end_sec is not None:
        if not _samples_in(left, proposal.start_sec, proposal.end_sec) or not _samples_in(
            right, proposal.start_sec, proposal.end_sec
        ):
            if left or right:
                return _done(base, "rejected", [predicate], ["lifetime_disjoint"])
    pair = _coincident(left, right, proposal.start_sec, proposal.end_sec)
    if pair is None:
        return _done(base, "uncertain", [predicate], ["missing_geometry"])
    sample_a, sample_b = pair
    if sample_a.frame != sample_b.frame:
        return _done(base, "rejected", [predicate], ["mixed_coordinate_frame"])
    metric_required = predicate in _METRIC
    if metric_required and not metric:
        return _done(base, "uncertain", [predicate], ["metric_scale_missing"])
    gate = _gate_reason(sample_a, metric_required=metric_required) or _gate_reason(
        sample_b, metric_required=metric_required
    )
    if gate is None and (not _series_feasible(left) or not _series_feasible(right)):
        gate = "uncertainty_gate"
    if gate is not None:
        return _done(base, "uncertain", [predicate], [gate])
    band = _band(proposal)
    if predicate == "distance_band" and band is None:
        return _done(base, "uncertain", [predicate], ["missing_distance_band"])
    holds = relation_holds(
        predicate,
        sample_a.center_m,
        sample_a.extent_m,
        sample_b.center_m,
        sample_b.extent_m,
        distance_band=band,
        subject_depth_m=sample_a.depth_m,
        object_depth_m=sample_b.depth_m,
        subject_motion_m=_motion(left, proposal.start_sec, proposal.end_sec),
    )
    if predicate == "visible" and not _track_active(tracks, subject, sample_a.t_sec):
        return _done(base, "uncertain", [predicate], ["missing_evidence"])
    rules = [predicate, "geometry"]
    if sample_a.residual_m is not None:
        rules.append("reprojection")
    if holds:
        return _done(base, "accepted", rules, [])
    reasons = ["geometry_contradiction"]
    if proposal.verification_status == "accepted":
        reasons.append("geometry_overrides_confidence")
    opposite = _OPPOSITE.get(predicate)
    if opposite and _opposite_edge(deltas, subject, obj, opposite, proposal):
        reasons.append("mutually_exclusive")
        rules.append("mutually_exclusive")
    return _done(base, "rejected", rules, reasons)


def _apply_decision(
    proposal: Proposal,
    resolution: Resolution,
    decision: ReviewDecision | None,
    *,
    failure: str | None,
    tracks: list[TrackRecord],
    samples: list[GeometrySample],
    events: list[TimelineEvent],
    aliases: dict[str, str],
) -> Resolution:
    if failure is not None or decision is None:
        reason = failure or "review_unavailable"
        return _done(resolution, "uncertain", [], [reason])
    reasons = list(decision.reasons)
    if decision.status == "corrected":
        resolution.predicate = decision.corrected_predicate or resolution.predicate
        return _done(resolution, "corrected", ["review"], reasons or ["corrected"])
    if decision.status != "accepted":
        return _done(resolution, decision.status, ["review"], reasons)
    if proposal.claim_kind == "action":
        if not _action_supported(proposal, resolution, events, aliases):
            return _done(resolution, "uncertain", ["review"], ["vlm_not_sole_evidence"])
        if not _semantic_evidence(resolution, tracks, samples):
            return _done(resolution, "uncertain", ["review"], ["missing_evidence"])
        evidence = _evidence_reason(resolution, tracks, samples)
        return _done(resolution, "accepted", ["review", "deterministic_event", evidence], reasons)
    if not _semantic_evidence(resolution, tracks, samples):
        return _done(resolution, "uncertain", ["review"], ["missing_evidence"])
    evidence = _evidence_reason(resolution, tracks, samples)
    return _done(resolution, "accepted", ["review", evidence], reasons)


def _review(
    reviewer: ReviewProvider,
    pending: list[tuple[Proposal, Resolution]],
    grouped: dict[str, list[GeometrySample]],
) -> tuple[str | None, list[ReviewDecision]]:
    if not pending:
        return None, []
    if getattr(reviewer, "failed", False):
        return None, []
    packets = [_packet(proposal, resolution, grouped) for proposal, resolution in pending]
    try:
        return None, list(reviewer.review(packets))
    except ReviewError as exc:
        return exc.code, []
    except Exception:
        return "malformed", []


def _packet(
    proposal: Proposal,
    resolution: Resolution,
    grouped: dict[str, list[GeometrySample]],
) -> ReviewPacket:
    measurements, counter = _facts(resolution.subject_id, resolution.object_id, grouped)
    return ReviewPacket(
        proposal_id=proposal.proposal_id,
        claim_kind=proposal.claim_kind,
        text=proposal.text,
        subject_id=resolution.subject_id,
        predicate=resolution.predicate,
        object_id=resolution.object_id,
        measurements=measurements or ["no geometric measurement settled this claim"],
        counter_evidence=counter or ["no counter-measurement was stored"],
    )


def _facts(
    subject: str | None,
    obj: str | None,
    grouped: dict[str, list[GeometrySample]],
) -> tuple[list[str], list[str]]:
    measurements: list[str] = []
    counter: list[str] = []
    for name, samples in (
        (subject, grouped.get(subject or "", [])),
        (obj, grouped.get(obj or "", [])),
    ):
        if not name or not samples:
            continue
        sample = samples[-1]
        measurements.append(
            f"{name} center_m {sample.center_m} extent_m {sample.extent_m} t_sec {sample.t_sec}"
        )
    if subject and obj and grouped.get(subject) and grouped.get(obj):
        left = grouped[subject][-1].center_m
        right = grouped[obj][-1].center_m
        if left[0] < right[0]:
            measurements.append(f"{subject} x is less than {obj} x")
            counter.append(f"right_of {subject} {obj} contradicts the stored x order")
        elif left[0] > right[0]:
            measurements.append(f"{subject} x is greater than {obj} x")
            counter.append(f"left_of {subject} {obj} contradicts the stored x order")
    return measurements[:8], counter[:8]


def _event_status(
    event: TimelineEvent,
    *,
    by_delta: dict[str, GraphDelta],
    grouped: dict[str, list[GeometrySample]],
    tracks: list[TrackRecord],
    metric: bool,
    root: Path,
) -> tuple[str, list[str], list[str]]:
    if event.verification.status != "accepted":
        return (
            event.verification.status,
            list(event.verification.rules),
            list(event.verification.reasons),
        )
    if not event.evidence or not _evidence_present(event, root):
        return "rejected", list(event.verification.rules), ["missing_evidence"]
    edge_deltas = [
        by_delta[ref] for ref in event.state_delta_refs if ref in by_delta and by_delta[ref].edge
    ]
    if edge_deltas:
        for delta in edge_deltas:
            edge = delta.edge
            assert edge is not None
            if edge.predicate in _METRIC and not metric:
                return "uncertain", [edge.predicate], ["metric_scale_missing"]
            pair = _coincident(
                grouped.get(edge.subject_id, []),
                grouped.get(edge.object_id, []),
                edge.start_sec,
                edge.end_sec,
            )
            if pair is None:
                return "uncertain", [edge.predicate], ["missing_geometry"]
            sample_a, sample_b = pair
            if _gate_reason(sample_a, metric_required=edge.predicate in _METRIC) or _gate_reason(
                sample_b, metric_required=edge.predicate in _METRIC
            ):
                return "uncertain", [edge.predicate], ["uncertainty_gate"]
            band = None
            if edge.distance_band is not None:
                band = (edge.distance_band.min_m, edge.distance_band.max_m)
            holds = relation_holds(
                edge.predicate,
                sample_a.center_m,
                sample_a.extent_m,
                sample_b.center_m,
                sample_b.extent_m,
                distance_band=band,
                subject_depth_m=sample_a.depth_m,
                object_depth_m=sample_b.depth_m,
                subject_motion_m=_motion(
                    grouped.get(edge.subject_id, []), edge.start_sec, edge.end_sec
                ),
            )
            if not holds:
                return "rejected", [edge.predicate, "geometry"], ["geometry_contradiction"]
        return "accepted", list(event.verification.rules), []
    if "track_count" in event.verification.rules:
        known = _track_ids(tracks)
        if any(participant not in known for participant in event.participants):
            return "rejected", ["track_count"], ["dangling_id"]
        return "accepted", ["track_count"], []
    return "uncertain", list(event.verification.rules), ["unrecomputed"]


def _done(base: Resolution, status: str, rules: list[str], reasons: list[str]) -> Resolution:
    base.status = status  # type: ignore[assignment]
    base.rules = rules
    base.reasons = reasons
    base.needs_review = False
    return base


def _bind(proposal: Proposal, nodes: set[str]) -> tuple[str | None, str | None]:
    if proposal.subject_id and proposal.object_id:
        return proposal.subject_id, proposal.object_id
    ordered = sorted(
        (node_id for node_id in nodes if node_id and node_id in proposal.text),
        key=proposal.text.find,
    )
    subject = proposal.subject_id or (ordered[0] if ordered else None)
    obj = proposal.object_id or (ordered[1] if len(ordered) > 1 else None)
    return subject, obj


def _parse_count(text: str) -> tuple[int, str] | None:
    parts = text.lower().replace("-", " ").split()
    if len(parts) < 2:
        return None
    if parts[0].isdigit():
        number = int(parts[0])
    elif parts[0] in _NUMBER_WORDS:
        number = _NUMBER_WORDS[parts[0]]
    else:
        return None
    label = parts[1].rstrip("s")
    if not label:
        return None
    return number, label


def _tracks_for_label(tracks: list[TrackRecord], label: str, at_sec: float) -> list[str]:
    latest: dict[str, TrackRecord] = {}
    for row in tracks:
        row_label = row.label_normalized.lower().rstrip("s")
        if row_label != label or row.t_sec > at_sec + 1e-6:
            continue
        current = latest.get(row.track_id)
        if current is None or row.t_sec >= current.t_sec:
            latest[row.track_id] = row
    return [track_id for track_id, row in latest.items() if row.state in _ACTIVE]


def _is_identity(proposal: Proposal) -> bool:
    text = proposal.text.lower()
    return proposal.predicate in {"same_as", "identity"} or "same object" in text


def _identity_link(tracks: list[TrackRecord], subject: str | None) -> bool:
    if subject is None:
        return False
    return any(row.track_id == subject and row.identity_link is not None for row in tracks)


def _index(samples: list[GeometrySample]) -> dict[str, list[GeometrySample]]:
    grouped: dict[str, list[GeometrySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.subject_id, []).append(sample)
    for rows in grouped.values():
        rows.sort(key=lambda sample: sample.t_sec)
    return grouped


def _node_ids(deltas: list[GraphDelta]) -> set[str]:
    return {delta.node.node_id for delta in deltas if delta.node is not None}


def _track_ids(tracks: list[TrackRecord]) -> set[str]:
    return {row.track_id for row in tracks}


def _samples_in(samples: list[GeometrySample], start: float, end: float) -> list[GeometrySample]:
    return [sample for sample in samples if start <= sample.t_sec < end]


def _coincident(
    left: list[GeometrySample],
    right: list[GeometrySample],
    start: float | None,
    end: float | None,
) -> tuple[GeometrySample, GeometrySample] | None:
    if start is not None and end is not None:
        left = _samples_in(left, start, end)
        right = _samples_in(right, start, end)
    if not left or not right:
        return None
    best = None
    best_gap = None
    for sample_a in left:
        for sample_b in right:
            gap = abs(sample_a.t_sec - sample_b.t_sec)
            if best_gap is None or gap < best_gap:
                best = (sample_a, sample_b)
                best_gap = gap
    return best


def _gate_reason(sample: GeometrySample, *, metric_required: bool) -> str | None:
    if metric_required and (sample.metric_scale != "metric" or not sample.calibration_id):
        return "metric_scale_missing"
    if sample.residual_m is not None and sample.residual_m > _RESIDUAL_MAX_M:
        return "uncertainty_gate"
    if (
        metric_required
        and sample.covariance_diag is not None
        and max(sample.covariance_diag) > _COVARIANCE_MAX
    ):
        return "uncertainty_gate"
    return None


def _series_feasible(samples: list[GeometrySample]) -> bool:
    series = [(sample.t_sec, sample.center_m) for sample in samples]
    return velocity_feasible(series)


def _band(proposal: Proposal) -> tuple[float, float] | None:
    match = _BAND.search(proposal.text)
    if match is None:
        return None
    low = float(match.group(1))
    high = float(match.group(2))
    if high <= low:
        return None
    return low, high


def _motion(samples: list[GeometrySample], start: float | None, end: float | None) -> float | None:
    window = samples
    if start is not None and end is not None:
        window = _samples_in(samples, start, end)
    if len(window) < 2:
        return None
    return _distance(window[0].center_m, window[-1].center_m)


def _distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) ** 0.5


def _opposite_edge(
    deltas: list[GraphDelta],
    subject: str,
    obj: str,
    predicate: str,
    proposal: Proposal,
) -> bool:
    for delta in deltas:
        edge = delta.edge
        if edge is None or edge.verification_status != "accepted":
            continue
        if (edge.subject_id, edge.predicate, edge.object_id) != (subject, predicate, obj):
            continue
        if proposal.start_sec is None or proposal.end_sec is None:
            return True
        if proposal.start_sec < edge.end_sec and edge.start_sec < proposal.end_sec:
            return True
    return False


def _track_active(tracks: list[TrackRecord], subject: str, t_sec: float) -> bool:
    rows = [row for row in tracks if row.track_id == subject and row.t_sec <= t_sec + 1e-6]
    if not rows:
        return False
    return max(rows, key=lambda row: row.t_sec).state in _ACTIVE


def _action_supported(
    proposal: Proposal,
    resolution: Resolution,
    events: list[TimelineEvent],
    aliases: dict[str, str],
) -> bool:
    """True when a deterministic accepted event already covers the action interval."""
    subject = resolution.subject_id
    if proposal.start_sec is None or proposal.end_sec is None or subject is None:
        return False
    names = {subject}
    if subject in aliases:
        names.add(aliases[subject])
    for event in events:
        if event.verification.status != "accepted" or not event.evidence:
            continue
        if proposal.start_sec >= event.end_sec or event.start_sec >= proposal.end_sec:
            continue
        participants = set(event.participants)
        participants.update(aliases.get(item, item) for item in event.participants)
        if names & participants:
            return True
    return False


def _participant_aliases(deltas: list[GraphDelta]) -> dict[str, str]:
    """Map a node id to its track id and the track id back to the node."""
    found: dict[str, str] = {}
    for delta in deltas:
        node = delta.node
        if node is None or node.track_id is None:
            continue
        found[node.node_id] = node.track_id
        found[node.track_id] = node.node_id
    return found


def _semantic_evidence(
    resolution: Resolution, tracks: list[TrackRecord], samples: list[GeometrySample]
) -> bool:
    ids = {item for item in (resolution.subject_id, resolution.object_id) if item}
    if any(row.track_id in ids for row in tracks):
        return True
    if any(sample.subject_id in ids for sample in samples):
        return True
    return bool(tracks or samples)


def _evidence_reason(
    resolution: Resolution, tracks: list[TrackRecord], samples: list[GeometrySample]
) -> str:
    ids = {item for item in (resolution.subject_id, resolution.object_id) if item}
    for row in tracks:
        if row.track_id in ids:
            return f"evidence:{row.observation_id}"
    for sample in samples:
        if sample.subject_id in ids:
            return f"evidence:{sample.sample_id}"
    if tracks:
        return f"evidence:{tracks[0].observation_id}"
    return f"evidence:{samples[0].sample_id}"


def _evidence_present(event: TimelineEvent, root: Path) -> bool:
    for evidence in event.evidence:
        refs = [evidence.geometry_ref, evidence.mask_ref]
        if not any(ref and (root / ref).is_file() for ref in refs):
            return False
    return True


def _sample_for_track(
    track_id: str, deltas: list[GraphDelta], samples: list[GeometrySample]
) -> GeometrySample | None:
    subjects = {track_id}
    for delta in deltas:
        if delta.node is not None and delta.node.track_id == track_id:
            subjects.add(delta.node.node_id)
    found = [sample for sample in samples if sample.subject_id in subjects]
    if not found:
        return None
    return sorted(found, key=lambda sample: sample.t_sec)[0]


def _geometry_ref(root: Path, sample: GeometrySample) -> str | None:
    directory = root / "geometry" / sample.subject_id
    if not directory.is_dir():
        return None
    needle = f'"sample_id":"{sample.sample_id}"'
    pretty = f'"sample_id": "{sample.sample_id}"'
    for path in sorted(directory.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        if needle in text or pretty in text:
            return path.relative_to(root).as_posix()
    return None


def _node_deltas(deltas: list[GraphDelta], track_ids: list[str]) -> list[str]:
    found = []
    for track_id in track_ids:
        for delta in deltas:
            if delta.node is not None and (
                delta.node.track_id == track_id or delta.node.node_id == track_id
            ):
                found.append(delta.delta_id)
                break
    return found


def _location(sample: GeometrySample, frame_name: str) -> Location3D:
    covariance = sample.covariance_diag or [0.01, 0.01, 0.01]
    return Location3D(
        frame=frame_name, center_m=list(sample.center_m), covariance_diag=list(covariance)
    )
