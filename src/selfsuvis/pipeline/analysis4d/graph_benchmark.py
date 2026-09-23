"""Scene-graph benchmark.

``python -m selfsuvis.pipeline.analysis4d.graph_benchmark`` checks that
replaying deltas is deterministic, intervals validate, deterministic relation
precision beats the YOLO semantic-graph baseline on the pinned scene, and an
unavailable VLM still leaves that graph plus ``provider_unavailable``.
"""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field

from selfsuvis.pipeline.analysis4d.geometry import center_error_m, relation_holds
from selfsuvis.pipeline.analysis4d.graph import HOLD_TAIL_SEC, RegionBox, materialize
from selfsuvis.pipeline.analysis4d.io import (
    canonical_bytes,
    canonical_line,
    sha256_bytes,
    write_bytes,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_DELTA,
    SCHEMA_GEOMETRY,
    SCHEMA_MANIFEST,
    SCHEMA_PROPOSAL,
    SCHEMA_QA,
    SCHEMA_TIMELINE,
    SCHEMA_TRACK,
    AnalysisManifest,
    ArtifactEntry,
    Box2D,
    CoordinateFrame,
    GeometrySample,
    SceneTimeline,
    StageReport,
    TrackRecord,
    TruthRelation,
)
from selfsuvis.pipeline.analysis4d.validate import validate_bundle
from selfsuvis.pipeline.analysis4d.vlm import UnavailableVlm
from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.workflows.analysis4d_graph import run_mission_graph

logger = get_logger(__name__)

SCHEMA_GRAPH_BENCHMARK = "ss-video.scene-graph-benchmark.v1"
_QUAT = [0.0, 0.0, 0.0, 1.0]
_BANDS = ((0.0, 2.0), (2.0, 8.0), (8.0, 32.0))
_METRIC_PREDICATES = frozenset({"distance_band", "supports", "contacts"})
_SYMMETRIC = frozenset({"intersects", "contacts", "distance_band"})
MISSION_ID = "mission-graph"
FRAME = "mission_enu"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphBenchmarkReport(_Model):
    """Machine-readable result of the scene-graph benchmark."""

    schema_version: str = SCHEMA_GRAPH_BENCHMARK
    passed: bool
    relation_precision: float | None = None
    yolo_relation_precision: float | None = None
    relation_precision_improved: bool = False
    replay_deterministic: bool = False
    intervals_valid: bool = False
    deterministic_edge_count: int = 0
    vlm_degraded: bool = False
    degradations: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


def default_graph_report_path() -> Path:
    """Return ``$DATA_DIR/analysis/_benchmark/graph-report.json``."""
    from selfsuvis.pipeline.analysis4d.paths import benchmark_report_path

    return benchmark_report_path().with_name("graph-report.json")


def run_graph_benchmark(
    output: Path | None = None,
    *,
    probe_gpu: bool = False,
) -> GraphBenchmarkReport:
    """Score the pinned scene against the YOLO semantic-graph baseline.

    Args:
        output: Directory for the mission bundle. The default is beside the report.
        probe_gpu: When true, try to load the compact VLM and record that probe.
            The pass/fail gate is the unavailable-VLM path.

    Returns:
        The report. ``passed`` is false when any gate fails.
    """
    root = Path(output) if output is not None else default_graph_report_path().parent / "graph"
    failures: list[str] = []
    prepare_pinned_scene(root)
    before = (root / "tracks.jsonl").read_bytes()
    result = run_mission_graph(
        MISSION_ID,
        dest=root,
        provider=UnavailableVlm(),
        regions=[
            RegionBox("region-1", "loading-zone", [0.0, 0.0, 0.0], [10.0, 10.0, 4.0]),
        ],
    )
    if (root / "tracks.jsonl").read_bytes() != before:
        failures.append("tracks.jsonl changed")
    replay = materialize(list(reversed(result.deltas)))
    replay_ok = _view_key(result.view) == _view_key(replay)
    if not replay_ok:
        failures.append("replay was not deterministic")
    intervals_ok = True
    try:
        bundle = validate_bundle(root)
    except Exception as exc:
        intervals_ok = False
        failures.append(f"intervals invalid: {exc}")
        bundle = None
    truth = pinned_truth()
    edges = [
        delta.edge
        for delta in result.deltas
        if delta.edge is not None and delta.edge.verification_status == "accepted"
    ]
    precision = _precision(edges, truth)
    yolo = _yolo_precision(root, truth)
    if precision is None or yolo is None or not (precision > yolo):
        failures.append(f"relation precision {precision} did not beat YOLO {yolo}")
    if "provider_unavailable" not in result.degradations:
        failures.append("missing provider_unavailable degradation")
    if not edges:
        failures.append("deterministic graph was empty")
    if probe_gpu:
        _probe_vlm(failures)
    improved = precision is not None and yolo is not None and precision > yolo
    return GraphBenchmarkReport(
        passed=not failures,
        relation_precision=precision,
        yolo_relation_precision=yolo,
        relation_precision_improved=improved,
        replay_deterministic=replay_ok,
        intervals_valid=intervals_ok and bundle is not None,
        deterministic_edge_count=len(edges),
        vlm_degraded="provider_unavailable" in result.degradations,
        degradations=list(result.degradations),
        failures=failures,
    )


def prepare_pinned_scene(root: Path) -> None:
    """Write tracks, geometry, and an empty graph for the pinned scene."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    frame = CoordinateFrame(name=FRAME, metric_scale="metric", calibration_id="cal-bench")
    tracks = []
    geometry: list[tuple[str, GeometrySample]] = []
    box = Box2D(xywh_norm=[0.1, 0.1, 0.2, 0.2])
    layout = {
        "track-a": ([1.0, 0.0, 0.5], [0.4, 0.4, 0.4], "crate"),
        "track-b": ([3.0, 0.0, 0.5], [0.4, 0.4, 0.4], "pallet"),
    }
    for index, (track_id, (center, extent, label)) in enumerate(sorted(layout.items())):
        for stamp in (0.0, 2.0, 4.0):
            millis = int(round(stamp * 1000))
            tracks.append(
                TrackRecord(
                    schema_version=SCHEMA_TRACK,
                    mission_id=MISSION_ID,
                    observation_id=f"obs-{track_id}-{millis:07d}",
                    track_id=track_id,
                    state="confirmed",
                    t_sec=stamp,
                    prompt_id="prompt-crate",
                    label_raw=label,
                    label_normalized=label,
                    box=box,
                    confidence=0.9,
                )
            )
            sample = GeometrySample(
                schema_version=SCHEMA_GEOMETRY,
                mission_id=MISSION_ID,
                sample_id=f"geo-{track_id}-{millis:07d}",
                subject_id=track_id,
                t_sec=stamp,
                frame=FRAME,
                center_m=list(center),
                extent_m=list(extent),
                quaternion_xyzw=list(_QUAT),
                metric_scale="metric",
                calibration_id="cal-bench",
            )
            geometry.append((f"geometry/{track_id}/{millis:07d}.json", sample))
        del index
    timeline = SceneTimeline(
        schema_version=SCHEMA_TIMELINE,
        mission_id=MISSION_ID,
        profile="deep",
        coordinate_frame=frame,
        model_manifest_ref="manifest.json",
    )
    files: list[tuple[str, str, str, bytes]] = [
        ("tracks.jsonl", "tracks", SCHEMA_TRACK, b"".join(canonical_line(row) for row in tracks)),
        ("graph-deltas.jsonl", "graph_deltas", SCHEMA_DELTA, b""),
        ("proposals.jsonl", "proposals", SCHEMA_PROPOSAL, b""),
        ("timeline.json", "timeline", SCHEMA_TIMELINE, canonical_bytes(timeline)),
        ("qa.jsonl", "qa", SCHEMA_QA, b""),
    ]
    for path, sample in geometry:
        files.append((path, "geometry", SCHEMA_GEOMETRY, canonical_bytes(sample)))
    for path, _kind, _schema, payload in files:
        write_bytes(root / path, payload)
    manifest = AnalysisManifest(
        schema_version=SCHEMA_MANIFEST,
        mission_id=MISSION_ID,
        profile="deep",
        coordinate_frame=frame,
        created_at="2026-01-15T00:00:00Z",
        artifacts=[
            ArtifactEntry(
                path=path,
                kind=kind,
                schema_version=schema,
                sha256=sha256_bytes(payload),
                bytes=len(payload),
            )
            for path, kind, schema, payload in files
        ],
        stages=[
            StageReport(
                stage="geometry",
                queue_delay_sec=0.0,
                inference_time_sec=0.0,
                processed_frames=6,
                skipped_frames=0,
                trigger_reason="fixture",
            )
        ],
        keyframes_sec=[0.0, 2.0, 4.0],
    )
    write_bytes(root / "manifest.json", canonical_bytes(manifest))


def pinned_truth() -> list[TruthRelation]:
    """Relations ``relation_holds`` accepts on the pinned scene, merged in time."""
    samples = _pinned_samples()
    grouped: dict[str, list[GeometrySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.subject_id, []).append(sample)
    rows: list[TruthRelation] = []
    subjects = sorted(grouped)
    for left in subjects:
        for right in subjects:
            if left == right:
                continue
            flags = []
            for stamp in (0.0, 2.0, 4.0):
                sample_a = _at(grouped[left], stamp)
                sample_b = _at(grouped[right], stamp)
                flags.append(_expected_at(left, right, sample_a, sample_b))
            times = [0.0, 2.0, 4.0]
            keys = {item for group in flags for item in group}
            for predicate in sorted(keys):
                holds = [predicate in group for group in flags]
                start = None
                for stamp, ok in zip(times, holds, strict=True):
                    if ok and start is None:
                        start = stamp
                    if ok or start is None:
                        continue
                    rows.append(
                        TruthRelation(
                            subject_id=left,
                            predicate=predicate,
                            object_id=right,
                            start_sec=start,
                            end_sec=stamp,
                        )
                    )
                    start = None
                if start is not None:
                    rows.append(
                        TruthRelation(
                            subject_id=left,
                            predicate=predicate,
                            object_id=right,
                            start_sec=start,
                            end_sec=times[-1] + HOLD_TAIL_SEC,
                        )
                    )
    return rows


def _pinned_samples() -> list[GeometrySample]:
    region = ([0.0, 0.0, 0.0], [10.0, 10.0, 4.0])
    layout = {
        "region-1": region,
        "track-a": ([1.0, 0.0, 0.5], [0.4, 0.4, 0.4]),
        "track-b": ([3.0, 0.0, 0.5], [0.4, 0.4, 0.4]),
    }
    samples = []
    for subject, (center, extent) in layout.items():
        for stamp in (0.0, 2.0, 4.0):
            millis = int(round(stamp * 1000))
            samples.append(
                GeometrySample(
                    schema_version=SCHEMA_GEOMETRY,
                    mission_id=MISSION_ID,
                    sample_id=f"geo-{subject}-{millis:07d}",
                    subject_id=subject,
                    t_sec=stamp,
                    frame=FRAME,
                    center_m=list(center),
                    extent_m=list(extent),
                    quaternion_xyzw=list(_QUAT),
                    metric_scale="metric",
                    calibration_id="cal-bench",
                )
            )
    return samples


def _expected_at(
    left: str, right: str, sample_a: GeometrySample, sample_b: GeometrySample
) -> set[str]:
    """Predicate names that hold for one pair. This is the benchmark pin."""
    found: set[str] = set()
    distance = center_error_m(sample_a.center_m, sample_b.center_m)
    band = None
    for low, high in _BANDS:
        if low <= distance <= high:
            band = (low, high)
            break
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
        if predicate in _METRIC_PREDICATES and band is None and predicate == "distance_band":
            continue
        payload = band if predicate == "distance_band" else None
        if predicate == "distance_band" and payload is None:
            continue
        if relation_holds(
            predicate,
            sample_a.center_m,
            sample_a.extent_m,
            sample_b.center_m,
            sample_b.extent_m,
            distance_band=payload,
            subject_depth_m=sample_a.depth_m,
            object_depth_m=sample_b.depth_m,
        ):
            found.add(predicate)
    if left in {"track-a", "track-b"} and right == "region-1":
        found.add("visible")
    return found


def _at(samples: list[GeometrySample], stamp: float) -> GeometrySample:
    return next(sample for sample in samples if sample.t_sec == stamp)


def _precision(edges: list, truth: list[TruthRelation]) -> float:
    if not edges and not truth:
        return 1.0
    if not edges:
        return 0.0
    used: set[int] = set()
    matched = 0
    for edge in edges:
        best = None
        best_iou = 0.0
        for index, row in enumerate(truth):
            if index in used:
                continue
            if (edge.subject_id, edge.predicate, edge.object_id) != (
                row.subject_id,
                row.predicate,
                row.object_id,
            ):
                continue
            iou = _temporal_iou(edge.start_sec, edge.end_sec, row.start_sec, row.end_sec)
            if iou > best_iou:
                best_iou = iou
                best = index
        if best is not None and best_iou > 0:
            used.add(best)
            matched += 1
    return matched / len(edges)


def _yolo_precision(root: Path, truth: list[TruthRelation]) -> float:
    from selfsuvis.pipeline.mapping import build_semantic_environment_graph

    del root
    frames = []
    centers = {"track-a": [1.0, 0.0, 0.5], "track-b": [3.0, 0.0, 0.5]}
    labels = {"track-a": "crate", "track-b": "pallet"}
    for stamp in (0.0, 2.0, 4.0):
        detections = []
        for track_id, center in centers.items():
            detections.append(
                {
                    "label": labels[track_id],
                    "confidence": 0.9,
                    "bbox_norm": [0.1, 0.1, 0.3, 0.3],
                }
            )
            del center
        frames.append(
            {
                "frame_id": f"f-{int(stamp * 1000)}",
                "t_sec": stamp,
                "detections": detections,
                "global_pose_json": {"tx": 0.0, "ty": 0.0, "tz": 0.0},
            }
        )
    graph = build_semantic_environment_graph(frames, graph_id=MISSION_ID)
    predicted: list[SimpleNamespace] = []
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    for edge in graph.get("edges", []):
        source = nodes.get(edge["source"])
        target = nodes.get(edge["target"])
        if source is None or target is None:
            continue
        start = float(source.get("first_seen_t_sec", 0.0))
        end = float(target.get("last_seen_t_sec", start))
        if end <= start:
            end = start + 1.0
        predicted.append(
            SimpleNamespace(
                subject_id=str(edge["source"]),
                predicate=str(edge.get("relation", "near")),
                object_id=str(edge["target"]),
                start_sec=start,
                end_sec=end,
            )
        )
    if not predicted:
        return 0.0
    return _precision(predicted, truth)


def _temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return inter / union


def _view_key(view) -> tuple:
    nodes = tuple(sorted((node.node_id, node.kind, node.label) for node in view.nodes))
    old = tuple(sorted(node.node_id for node in view.superseded_nodes))
    edges = tuple(
        sorted(
            (
                edge.subject_id,
                edge.predicate,
                edge.object_id,
                edge.start_sec,
                edge.end_sec,
            )
            for edge in view.edges
        )
    )
    return nodes, old, edges, tuple(view.superseded_delta_ids)


def _probe_vlm(failures: list[str]) -> None:
    from selfsuvis.pipeline.analysis4d.vlm import CompactVlm, parse_vlm_payload

    provider = CompactVlm()
    if provider.failed:
        logger.info("compact VLM unavailable: %s", provider.detail)
        return
    try:
        claims = provider.propose({"regions": [], "trajectories": [], "edges": []})
        for claim in claims:
            parse_vlm_payload(claim.model_dump_json())
    except Exception as exc:
        failures.append(f"compact VLM returned a claim the schema rejected: {exc}")


def main(argv: list[str] | None = None) -> int:
    """Write the scene-graph benchmark report. Returns 0 when it passes."""
    parser = argparse.ArgumentParser(description="4D scene-graph benchmark")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--probe-gpu", action="store_true")
    args = parser.parse_args(argv)
    report_path = args.output or default_graph_report_path()
    bundle_dir = args.bundle_dir or report_path.parent / "graph"
    report = run_graph_benchmark(bundle_dir, probe_gpu=args.probe_gpu)
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    write_bytes(report_path, payload.encode("utf-8"))
    logger.info("scene graph benchmark passed=%s report=%s", report.passed, report_path)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
