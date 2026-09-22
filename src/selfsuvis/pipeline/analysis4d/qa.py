"""Executable graph programs for spatial Video-QA.

A program runs against accepted events and edges only. Empty and ambiguous
results are dropped. The answer is the program result. Causal programs are
not generated and do not execute.
"""

import re
from dataclasses import dataclass

from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_QA,
    GraphEdge,
    QaAnswer,
    QaPair,
    QaRecord,
    TimelineEvent,
    TrackRecord,
)

_ENTERED = re.compile(r"^entered\(\?track,\s*([A-Za-z0-9_.:-]+),\s*after=([0-9]+(?:\.[0-9]+)?)\)$")
_COUNT = re.compile(r"^count\(([A-Za-z0-9_-]+),\s*at=([0-9]+(?:\.[0-9]+)?)\)$")
_EVIDENCE = re.compile(r"^evidence\(([A-Za-z0-9_.:-]+)\)$")
_LEFT = re.compile(r"^left_of\(\?subject,\s*([A-Za-z0-9_.:-]+)\)$")
_STATE = re.compile(
    r"^(picked_up|put_down|left_region|approached)\(\?track,\s*([A-Za-z0-9_.:-]+)\)$"
)


@dataclass(frozen=True)
class ProgramOutcome:
    """One unambiguous answer and the events that support it."""

    qa_type: str
    question: str
    answer: QaAnswer
    interval_sec: list[float]
    evidence_event_ids: list[str]


def execute_graph_program(
    program: str,
    *,
    events: list[TimelineEvent],
    tracks: list[TrackRecord],
    edges: list[GraphEdge],
) -> ProgramOutcome | None:
    """Execute one graph program.

    Args:
        program: A spatial, temporal, count, state-change, or evidence program.
        events: Timeline events. Only ``accepted`` rows are evidence.
        tracks: Track labels used by count programs.
        edges: Accepted edges used by ``left_of``.

    Returns:
        The answer, or None when the program is empty, ambiguous, or causal.
    """
    accepted = [event for event in events if event.verification.status == "accepted"]
    if program.startswith("caused(") or program.startswith("causal("):
        return None
    entered = _ENTERED.fullmatch(program)
    if entered:
        return _entered(entered.group(1), float(entered.group(2)), accepted)
    counted = _COUNT.fullmatch(program)
    if counted:
        return _count(counted.group(1), float(counted.group(2)), accepted, tracks)
    evidence = _EVIDENCE.fullmatch(program)
    if evidence:
        return _evidence(evidence.group(1), accepted)
    left = _LEFT.fullmatch(program)
    if left:
        return _left_of(left.group(1), accepted, edges)
    state = _STATE.fullmatch(program)
    if state:
        return _state(state.group(1), state.group(2), accepted)
    return None


def generate_qa(
    mission_id: str,
    events: list[TimelineEvent],
    tracks: list[TrackRecord],
    edges: list[GraphEdge],
) -> list[QaRecord]:
    """Build QA rows from programs that have one accepted answer.

    Args:
        mission_id: Mission id stored on each log row.
        events: Verified timeline events.
        tracks: Track labels.
        edges: Accepted graph edges.

    Returns:
        QA records. Zero rows is valid when every program is empty or ambiguous.
    """
    programs = _programs(events, edges)
    records: list[QaRecord] = []
    seen: set[str] = set()
    for program in programs:
        outcome = execute_graph_program(program, events=events, tracks=tracks, edges=edges)
        if outcome is None or not outcome.evidence_event_ids:
            continue
        qa_id = _qa_id(program)
        if qa_id in seen:
            continue
        seen.add(qa_id)
        records.append(
            QaRecord(
                schema_version=SCHEMA_QA,
                mission_id=mission_id,
                qa_id=qa_id,
                type=outcome.qa_type,
                question=outcome.question,
                answer=outcome.answer,
                graph_program=program,
                interval_sec=outcome.interval_sec,
                evidence_event_ids=outcome.evidence_event_ids,
                verification_status="accepted",
            )
        )
    return records


def qa_pairs(records: list[QaRecord]) -> list[QaPair]:
    """Project log rows onto the timeline's embedded QA pairs."""
    return [
        QaPair(
            qa_id=record.qa_id,
            type=record.type,
            question=record.question,
            answer=record.answer,
            graph_program=record.graph_program,
            interval_sec=list(record.interval_sec),
            evidence_event_ids=list(record.evidence_event_ids),
            verification_status=record.verification_status,
        )
        for record in records
    ]


def publishable_qa(records: list[QaRecord]) -> list[QaRecord]:
    """Return accepted answers that cite an event. Rejected rows stay out."""
    return [
        record
        for record in records
        if record.verification_status == "accepted" and record.evidence_event_ids
    ]


def _programs(events: list[TimelineEvent], edges: list[GraphEdge]) -> list[str]:
    accepted = [event for event in events if event.verification.status == "accepted"]
    programs: list[str] = []
    by_region: dict[str, list[TimelineEvent]] = {}
    for event in accepted:
        if event.type != "entered_region" or len(event.participants) < 2:
            continue
        region = event.participants[-1]
        by_region.setdefault(region, []).append(event)
    for region, rows in sorted(by_region.items()):
        if len(rows) == 1:
            programs.append(f"entered(?track, {region}, after={_num(rows[0].start_sec)})")
    labels: set[str] = set()
    for event in accepted:
        if event.type != "count_changed":
            continue
        label = event.summary.split(" ", 1)[0]
        if not label or label in labels:
            continue
        labels.add(label)
        programs.append(f"count({label}, at={_num(event.start_sec)})")
    for event_type in ("picked_up", "put_down", "left_region", "approached"):
        rows = [event for event in accepted if event.type == event_type]
        if len(rows) == 1 and len(rows[0].participants) >= 2:
            other = rows[0].participants[-1]
            programs.append(f"{event_type}(?track, {other})")
    objects = sorted(
        {
            edge.object_id
            for edge in edges
            if edge.predicate == "left_of" and edge.verification_status == "accepted"
        }
    )
    for object_id in objects:
        programs.append(f"left_of(?subject, {object_id})")
    evidenced = [
        event
        for event in accepted
        if any(item.geometry_ref or item.mask_ref for item in event.evidence)
    ]
    if evidenced:
        earliest = min(evidenced, key=lambda event: (event.start_sec, event.event_id))
        programs.append(f"evidence({earliest.event_id})")
    return programs


def _entered(region: str, after: float, events: list[TimelineEvent]) -> ProgramOutcome | None:
    matches = []
    for event in events:
        if event.type != "entered_region" or event.start_sec + 1e-9 < after:
            continue
        if region not in event.participants:
            continue
        tracks = [item for item in event.participants if item != region]
        if len(tracks) == 1:
            matches.append((tracks[0], event))
    return _one_track(
        matches,
        qa_type="spatial",
        question=f"Which tracked object entered {region} after {_num(after)} seconds?",
    )


def _count(
    label: str,
    at_sec: float,
    events: list[TimelineEvent],
    tracks: list[TrackRecord],
) -> ProgramOutcome | None:
    support = [
        event
        for event in events
        if event.type == "count_changed"
        and abs(event.start_sec - at_sec) <= 1e-6
        and event.summary.startswith(label + " ")
    ]
    if len(support) != 1:
        return None
    event = support[0]
    labels = {
        row.track_id: row.label_normalized.lower().rstrip("s")
        for row in tracks
        if row.track_id in event.participants
    }
    if any(labels.get(track_id) != label for track_id in event.participants):
        return None
    return ProgramOutcome(
        qa_type="count",
        question=f"How many {label} tracks are active at {_num(at_sec)} seconds?",
        answer=QaAnswer(kind="count", value=len(event.participants), unit=None),
        interval_sec=[event.start_sec, event.end_sec],
        evidence_event_ids=[event.event_id],
    )


def _evidence(event_id: str, events: list[TimelineEvent]) -> ProgramOutcome | None:
    matches = [event for event in events if event.event_id == event_id and event.evidence]
    if len(matches) != 1:
        return None
    event = matches[0]
    ref = event.evidence[0].geometry_ref or event.evidence[0].mask_ref or event.evidence[0].frame_id
    return ProgramOutcome(
        qa_type="evidence",
        question=f"Where is the stored evidence for {event_id}?",
        answer=QaAnswer(kind="text", value=ref, unit=None),
        interval_sec=[event.start_sec, event.end_sec],
        evidence_event_ids=[event.event_id],
    )


def _left_of(
    object_id: str, events: list[TimelineEvent], edges: list[GraphEdge]
) -> ProgramOutcome | None:
    subjects = []
    for edge in edges:
        if (
            edge.verification_status == "accepted"
            and edge.predicate == "left_of"
            and edge.object_id == object_id
        ):
            subjects.append((edge.subject_id, edge))
    unique = {subject for subject, _edge in subjects}
    if len(unique) != 1:
        return None
    subject = next(iter(unique))
    support = [event for event in events if subject in event.participants and event.evidence]
    if not support:
        return None
    event = min(support, key=lambda item: (item.start_sec, item.event_id))
    edge = subjects[0][1]
    return ProgramOutcome(
        qa_type="spatial",
        question=f"Which tracked object is left of {object_id}?",
        answer=QaAnswer(kind="track_ref", value=subject, unit=None),
        interval_sec=[edge.start_sec, edge.end_sec],
        evidence_event_ids=[event.event_id],
    )


def _state(event_type: str, other: str, events: list[TimelineEvent]) -> ProgramOutcome | None:
    matches = []
    for event in events:
        if event.type != event_type or other not in event.participants:
            continue
        tracks = [item for item in event.participants if item != other]
        if len(tracks) == 1:
            matches.append((tracks[0], event))
    question = f"Which tracked object {event_type.replace('_', ' ')} {other}?"
    return _one_track(matches, qa_type="state_change", question=question)


def _one_track(
    matches: list[tuple[str, TimelineEvent]],
    *,
    qa_type: str,
    question: str,
) -> ProgramOutcome | None:
    if len({track for track, _event in matches}) != 1 or not matches:
        return None
    track, event = matches[0]
    if not event.evidence:
        return None
    return ProgramOutcome(
        qa_type=qa_type,
        question=question,
        answer=QaAnswer(kind="track_ref", value=track, unit=None),
        interval_sec=[event.start_sec, event.end_sec],
        evidence_event_ids=[event.event_id],
    )


def _qa_id(program: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", program).strip("-").lower()
    return f"qa-{slug[:80]}"


def _num(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"
