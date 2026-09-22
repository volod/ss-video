"""Publish accepted 4D events as versioned site envelopes.

The message body is the ODCS contract ``verified-scene-event`` 1.0.0. The
site handoff is the existing ss-common ``event-envelope`` 1.0.0, so fusion-rt
ingests it through the current event path. Rejected and uncertain rows are
not written. A second call with the same event id does not append again.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from selfsuvis.pipeline.analysis4d.schemas import SceneTimeline
from selfsuvis.pipeline.analysis4d.verified_scene_event import VerifiedSceneEvent
from ss_contracts.models import EventEnvelope

LEDGER_NAME = "published-events.jsonl"


def publish_accepted(
    dest: Path,
    timeline: SceneTimeline,
    *,
    zone_id: str,
    sensor_id: str,
    published_at: datetime | None = None,
) -> list[str]:
    """Append envelopes for accepted events that are not already in the ledger.

    Args:
        dest: Mission 4D directory. The ledger is ``published-events.jsonl``.
        timeline: Materialized timeline. Only ``accepted`` rows are eligible.
        zone_id: Site zone stored on the envelope.
        sensor_id: Reporting sensor stored on the envelope.
        published_at: Envelope time. The default is the current UTC time.

    Returns:
        Event ids appended by this call. Already published ids are omitted.
    """
    stamp = published_at or datetime.now(UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    seen = set(_published_ids(dest))
    fresh: list[str] = []
    lines: list[str] = []
    for event in timeline.events:
        if event.verification.status != "accepted" or event.event_id in seen:
            continue
        contract = VerifiedSceneEvent(
            event_id=event.event_id,
            mission_id=timeline.mission_id,
            profile=timeline.profile,
            event_type=event.type,
            summary=event.summary,
            start_sec=event.start_sec,
            end_sec=event.end_sec,
            participants=list(event.participants),
            confidence=event.confidence,
            verification_status="accepted",
            supersedes=event.supersedes,
            artifact_uri="timeline.json",
            published_at=stamp,
        )
        envelope = EventEnvelope(
            ts=stamp,
            zone_id=zone_id,
            sensor_id=sensor_id,
            confidence=event.confidence,
            payload=contract.model_dump(mode="json"),
            artifact_uri="timeline.json",
        )
        VerifiedSceneEvent.model_validate(envelope.payload)
        encoded = envelope.model_dump_json()
        EventEnvelope.model_validate_json(encoded)
        lines.append(encoded + "\n")
        fresh.append(event.event_id)
        seen.add(event.event_id)
    if lines:
        path = Path(dest) / LEDGER_NAME
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
    return fresh


def published_ids(dest: Path) -> list[str]:
    """Return event ids already written to the mission ledger, in append order."""
    return _published_ids(dest)


def _published_ids(dest: Path) -> list[str]:
    path = Path(dest) / LEDGER_NAME
    if not path.is_file():
        return []
    found: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line).get("payload") or {}
        event_id = payload.get("event_id")
        if event_id and event_id not in found:
            found.append(str(event_id))
    return found
