"""Replay saved local-run artifacts as ordered realtime event envelopes."""

import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from selfsuvis.fusion_rt.events import NodeHealthEvent, SensorEvent, ThreatEvent
from selfsuvis.pipeline.core.freshness import apply_freshness
from selfsuvis.pipeline.fusion.sectors import (
    build_route_segment_id,
    sectorize_global_positions,
    unique_sector_sequence,
)
from selfsuvis.pipeline.media.bridge_common import flatten_packet_batches
from selfsuvis.pipeline.media.drone_bridge import bridge_mavlink_messages
from selfsuvis.pipeline.media.ros_bridge import ros_message_to_packets

from .ingest import normalize_packets


def _event_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["event_time"],
        row["event_kind"],
        row["sector_id"],
        row["sensor_type"],
    )


def _event_ingest_delay_sec(event: dict[str, Any], delays: dict[str, float]) -> float:
    delay = float(delays.get(str(event.get("event_kind", "")), 0.0) or 0.0)
    delay += float(delays.get(str(event.get("sensor_type", "")), 0.0) or 0.0)
    return delay


def _trace_packets(records: list[dict[str, Any]], *, backend: str) -> list[dict[str, Any]]:
    if backend == "mavlink":
        return bridge_mavlink_messages(records)
    if backend == "ros":
        return flatten_packet_batches(records, ros_message_to_packets)
    raise ValueError(f"unsupported replay backend: {backend}")


def replay_local_run(
    output_dir: Path,
    *,
    node_id_prefix: str = "node",
    ingest_start: datetime | None = None,
    ingest_delay_sec_by_kind: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    ingest_start = ingest_start or datetime.now(timezone.utc)
    ingest_delay_sec_by_kind = dict(ingest_delay_sec_by_kind or {})

    video_dirs = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_dir() and (path / "local_threat_assessment.json").exists()
    )
    raw_events: list[dict[str, Any]] = []
    for index, video_dir in enumerate(video_dirs, start=1):
        raw_events.extend(_video_events(video_dir, node_id=f"{node_id_prefix}_{index}"))

    raw_events.sort(key=_event_sort_key)
    replayed: list[dict[str, Any]] = []
    for idx, event in enumerate(raw_events):
        event_dt = _parse_iso(event["event_time"])
        base_ingest = ingest_start + timedelta(seconds=idx * 0.25)
        extra_delay = _event_ingest_delay_sec(event, ingest_delay_sec_by_kind)
        ingest_time = max(base_ingest, event_dt) + timedelta(seconds=extra_delay)
        enriched = dict(event)
        enriched["ingest_time"] = ingest_time.isoformat()
        replayed.append(apply_freshness(enriched))
    return replayed


def write_replay_jsonl(events: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(event), sort_keys=True) for event in events]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        record = json.loads(text)
        if isinstance(record, dict):
            records.append(record)
    return records


def replay_bridge_trace(
    trace_path: Path,
    *,
    backend: str,
) -> list[dict[str, Any]]:
    backend_name = str(backend or "").strip().lower()
    records = load_jsonl_records(trace_path)
    return normalize_packets(_trace_packets(records, backend=backend_name))


def _video_events(video_dir: Path, *, node_id: str) -> list[dict[str, Any]]:
    local_threat = _load_json(video_dir / "local_threat_assessment.json")
    primitives = _load_json(video_dir / "threat_primitives.json")
    physical_state = _load_json(video_dir / "physical_state_summary.json")
    full_fusion = _load_json(video_dir / "full_state_fusion.json")
    field_state = _load_json(video_dir / "field_state_summary.json")
    sector_ids, time_range_sec = _sector_index(full_fusion, video_dir.name)
    route_id = build_route_segment_id(video_dir.name, sector_ids)
    start_dt = _base_event_time(video_dir.name, time_range_sec[0])
    end_dt = _base_event_time(video_dir.name, time_range_sec[1])
    primary_sector = sector_ids[0] if sector_ids else "unknown"

    events: list[dict[str, Any]] = []
    events.append(
        SensorEvent(
            event_time=start_dt.isoformat(),
            ingest_time=start_dt.isoformat(),
            node_id=node_id,
            sensor_type="fusion",
            sector_id=primary_sector,
            payload={
                "video_id": video_dir.name,
                "route_id": route_id,
                "physical_state": physical_state,
                "field_state": field_state.get("clip_level_fields", {}),
            },
        ).to_dict()
    )

    for primitive in list(primitives.get("primitives") or []):
        events.append(
            ThreatEvent(
                event_time=start_dt.isoformat(),
                ingest_time=start_dt.isoformat(),
                node_id=node_id,
                sensor_type=str(primitive.get("type", "threat")),
                sector_id=primary_sector,
                payload={
                    "video_id": video_dir.name,
                    "route_id": route_id,
                    "threat_type": primitive.get("type"),
                    "score": primitive.get("score", 0.0),
                    "uncertainty": primitive.get("uncertainty", 0.0),
                    "evidence_sources": list(primitive.get("evidence_sources") or []),
                },
            ).to_dict()
        )

    events.append(
        ThreatEvent(
            event_time=end_dt.isoformat(),
            ingest_time=end_dt.isoformat(),
            node_id=node_id,
            sensor_type="local_threat",
            sector_id=primary_sector,
            payload={
                "video_id": video_dir.name,
                "route_id": route_id,
                "local_threat_score": local_threat.get("local_threat_score", 0.0),
                "automation_confidence": local_threat.get("automation_confidence", 1.0),
                "trust_penalty": local_threat.get("trust_penalty", 0.0),
                "recommended_action": local_threat.get("recommended_action", "continue"),
                "top_threats": list(local_threat.get("top_threats") or []),
            },
        ).to_dict()
    )

    events.append(
        NodeHealthEvent(
            event_time=end_dt.isoformat(),
            ingest_time=end_dt.isoformat(),
            node_id=node_id,
            sensor_type="analytics",
            sector_id=primary_sector,
            payload={
                "video_id": video_dir.name,
                "route_id": route_id,
                "automation_confidence": local_threat.get("automation_confidence", 1.0),
                "trust_penalty": local_threat.get("trust_penalty", 0.0),
                "disagreement_count": local_threat.get("disagreement_count", 0),
                "outage_sec": 0.0,
                "model_name": "local_pipeline",
            },
        ).to_dict()
    )
    return events


def _sector_index(full_fusion: dict[str, Any], video_name: str) -> tuple[list[str], list[float]]:
    platform = full_fusion.get("platform") or {}
    origin = platform.get("origin_lla") or {}
    smoothed = (full_fusion.get("map_state") or {}).get("smoothed_samples") or []
    positions = [
        dict(row.get("position_enu_m") or {}) for row in smoothed if row.get("position_enu_m")
    ]
    if origin and positions:
        sectors = unique_sector_sequence(
            sectorize_global_positions(origin, positions, tile_size_m=50.0)
        )
    else:
        sectors = []
    if smoothed:
        time_range = [
            float(smoothed[0].get("t_sec", 0.0) or 0.0),
            float(smoothed[-1].get("t_sec", 0.0) or 0.0),
        ]
    else:
        time_range = [0.0, 0.0]
    return sectors, time_range


def _base_event_time(video_name: str, t_sec: float) -> datetime:
    seed = sum(ord(ch) for ch in video_name) % 86_400
    base = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seed)
    return base + timedelta(seconds=float(t_sec or 0.0))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_iso(value: str) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
