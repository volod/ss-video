"""Write 3D observations for an existing 2D track bundle.

A depth-provider failure records a degradation and leaves ``tracks.jsonl``
unchanged. Relative depth is not stored as metric. Dynamic points stay out of
the static cloud.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from selfsuvis.pipeline.analysis4d.appearance import Descriptor, PrototypeBank
from selfsuvis.pipeline.analysis4d.camera import (
    CameraModel,
    apply_perspective_fields,
    resolve_scale,
)
from selfsuvis.pipeline.analysis4d.geometry_providers import DepthMap, GeometryView
from selfsuvis.pipeline.analysis4d.io import (
    canonical_bytes,
    read_jsonl,
    read_model,
    sha256_bytes,
    write_bytes,
)
from selfsuvis.pipeline.analysis4d.paths import analysis_dir
from selfsuvis.pipeline.analysis4d.reconstruct import (
    StaticCloud,
    backproject_mask,
    compensate_motion,
    fit_gravity_box,
    normals_from_depth,
    transform_normals,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_EMBEDDING,
    SCHEMA_GEOMETRY,
    AnalysisManifest,
    ArtifactEntry,
    CoordinateFrame,
    Degradation,
    GeometrySample,
    SceneTimeline,
    StageReport,
    TrackEmbedding,
    TrackRecord,
)
from selfsuvis.pipeline.analysis4d.store import AnalysisStore
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

_MOTION_SPEED = 0.15
_ACTIVE = frozenset({"tentative", "confirmed", "occluded"})


@dataclass
class GeometryResult:
    """What one mission reconstruction wrote."""

    dest: Path
    samples: list[GeometrySample] = field(default_factory=list)
    static_ids: list[str] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    tracks_sha256: str = ""


def run_mission_geometry(
    mission_id: str,
    views: list[GeometryView],
    *,
    dest: Path | None = None,
    depth: object,
    appearance: object | None = None,
    calibration: object | None = None,
    static_cloud: StaticCloud | None = None,
) -> GeometryResult:
    """Back-project eligible tracks and write geometry plus embeddings.

    Args:
        mission_id: Mission whose ``tracks.jsonl`` already exists.
        views: Per-timestamp cameras and optional images, masks, and features.
        dest: Artifact directory. The default is the mission 4D directory.
        depth: Depth provider. A failure preserves the track file.
        appearance: Optional masked descriptor provider.
        calibration: Optional Perspective Fields-style fallback.
        static_cloud: Receives static points. Dynamic tracks are omitted.

    Returns:
        Written samples and the track-file digest, which matches the digest
        from before this call.
    """
    target = Path(dest) if dest is not None else analysis_dir(mission_id)
    tracks_path = target / "tracks.jsonl"
    before = tracks_path.read_bytes() if tracks_path.is_file() else b""
    records = read_jsonl(tracks_path, TrackRecord) if tracks_path.is_file() else []
    cloud = static_cloud if static_cloud is not None else StaticCloud()
    by_time = {_stamp(view.t_sec): view for view in views}
    degradations: list[Degradation] = []
    samples: list[GeometrySample] = []
    embedding_files: list[tuple[str, bytes]] = []
    if getattr(depth, "failed", False):
        degradations.append(
            Degradation(
                code="provider_unavailable",
                stage="geometry",
                detail="depth provider did not load; 2D tracks were kept",
            )
        )
    else:
        samples, embedding_files, degradations = _reconstruct(
            mission_id,
            records,
            by_time,
            depth=depth,
            appearance=appearance,
            calibration=calibration,
            cloud=cloud,
            dest=target,
        )
    _write_samples(target, samples)
    for path, payload in embedding_files:
        write_bytes(target / path, payload)
    _update_manifest(target, samples, embedding_files, degradations)
    after = tracks_path.read_bytes() if tracks_path.is_file() else b""
    if after != before:
        raise RuntimeError("geometry reconstruction modified tracks.jsonl")
    return GeometryResult(
        dest=target,
        samples=samples,
        static_ids=sorted(cloud.points),
        degradations=[item.code for item in degradations],
        tracks_sha256=sha256_bytes(before),
    )


def _reconstruct(
    mission_id: str,
    records: list[TrackRecord],
    views: dict[float, GeometryView],
    *,
    depth: object,
    appearance: object | None,
    calibration: object | None,
    cloud: StaticCloud,
    dest: Path,
) -> tuple[list[GeometrySample], list[tuple[str, bytes]], list[Degradation]]:
    grouped: dict[str, list[TrackRecord]] = {}
    for record in records:
        if record.state not in _ACTIVE:
            continue
        grouped.setdefault(record.track_id, []).append(record)
    samples: list[GeometrySample] = []
    files: list[tuple[str, bytes]] = []
    flags: set[str] = set()
    produced = 0
    attempted = 0
    bank = PrototypeBank()
    frame_name = _frame_name(dest)
    for track_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: row.t_sec)
        dynamic = _dynamic(ordered)
        collected: list[tuple[float, np.ndarray, TrackRecord, CameraModel, str, np.ndarray]] = []
        for record in ordered:
            view = views.get(_stamp(record.t_sec))
            if view is None:
                continue
            attempted += 1
            camera, depth_map, scale = _observe(view, depth, calibration, flags)
            if camera is None or depth_map is None or scale == "unavailable":
                continue
            mask = (
                view.mask
                if view.mask is not None
                else _mask_from_box(
                    record.box.xywh_norm, depth_map.values.shape[1], depth_map.values.shape[0]
                )
            )
            points, _pixels, z = backproject_mask(depth_map.values, mask, camera)
            if len(points) == 0:
                flags.add("empty_scene")
                continue
            _ = _normals(depth_map, camera)
            collected.append((record.t_sec, points, record, camera, scale, z))
            if appearance is not None and not getattr(appearance, "failed", False):
                described = _describe(appearance, view, mask)
                if described is not None:
                    prototype = bank.update(track_id, described)
                    files.append(_embedding_file(mission_id, track_id, described, "observation"))
                    files.append(_embedding_file(mission_id, track_id, prototype, "prototype"))
        window = [(stamp, pts) for stamp, pts, _record, _camera, _scale, _z in collected]
        for stamp, points, record, camera, scale, z in collected:
            fused = compensate_motion(window, dynamic=dynamic)
            fitted = fit_gravity_box(
                fused if dynamic else points, metric_scale=scale, dynamic=dynamic
            )
            if fitted is None:
                continue
            cloud.add(track_id, points, dynamic=dynamic)
            sample = _sample(mission_id, frame_name, record, camera, fitted, z, scale)
            samples.append(sample)
            produced += 1
            if scale == "relative":
                flags.add("relative_depth_only")
    if attempted and produced == 0 and "provider_unavailable" not in flags:
        flags.add("provider_unavailable")
    if not records:
        flags.add("empty_scene")
    degradations = [
        Degradation(code=code, stage="geometry", detail=_detail(code)) for code in sorted(flags)
    ]
    return samples, _dedupe_embeddings(files), degradations


def _observe(
    view: GeometryView,
    depth: object,
    calibration: object | None,
    flags: set[str],
) -> tuple[CameraModel | None, DepthMap | None, str]:
    camera = view.camera
    if camera is None or not camera.intrinsics_ok or not camera.pose_known:
        estimate = _fields(calibration, view)
        if estimate is None:
            flags.add(
                "calibration_missing"
                if camera is None or not camera.intrinsics_ok
                else "pose_missing"
            )
            if camera is not None and not camera.pose_known:
                flags.add("pose_missing")
        else:
            width, height = _image_size(view, camera)
            camera = apply_perspective_fields(estimate, width=width, height=height, camera=camera)
            if not camera.pose_known:
                flags.add("pose_missing")
    try:
        depth_map = depth.estimate(view)
    except Exception:
        logger.warning("depth estimate failed at t=%s", _stamp(view.t_sec))
        flags.add("provider_unavailable")
        return camera, None, "unavailable"
    if depth_map is None:
        flags.add("provider_unavailable")
        return camera, None, "unavailable"
    scale = resolve_scale(getattr(depth, "scale", depth_map.scale), camera)
    if getattr(depth, "scale", "") == "metric" and scale != "metric":
        flags.add("metric_scale_missing")
    if scale == "unavailable":
        flags.add(
            "pose_missing" if camera is None or not camera.pose_known else "calibration_missing"
        )
    return camera, depth_map, scale


def _sample(
    mission_id: str,
    frame_name: str,
    record: TrackRecord,
    camera: CameraModel,
    fitted,
    z: np.ndarray,
    scale: str,
) -> GeometrySample:
    millis = int(round(record.t_sec * 1000))
    center = camera.center_m()
    mean_extent = sum(fitted.extent_m) / 3.0
    base = 0.85 if scale == "metric" else 0.4
    confidence = max(0.0, min(1.0, base * math.exp(-fitted.residual_m / max(mean_extent, 1e-3))))
    finite_z = z[np.isfinite(z)]
    depth_m = float(np.median(finite_z)) if finite_z.size else None
    return GeometrySample(
        schema_version=SCHEMA_GEOMETRY,
        mission_id=mission_id,
        sample_id=f"geo-{record.track_id}-{millis:07d}",
        subject_id=record.track_id,
        t_sec=record.t_sec,
        frame=frame_name,
        center_m=list(fitted.center_m),
        extent_m=list(fitted.extent_m),
        quaternion_xyzw=list(fitted.quaternion_xyzw),
        depth_m=depth_m,
        pose_position_m=[float(center[0]), float(center[1]), float(center[2])],
        metric_scale=scale,
        calibration_id=camera.calibration_id if scale == "metric" else None,
        covariance_diag=list(fitted.covariance_diag),
        residual_m=fitted.residual_m,
        observation_count=fitted.observation_count,
        scale_confidence=confidence,
        dynamic=fitted.dynamic,
    )


def _write_samples(dest: Path, samples: list[GeometrySample]) -> None:
    for sample in samples:
        millis = int(round(sample.t_sec * 1000))
        path = dest / "geometry" / sample.subject_id / f"{millis:07d}.json"
        write_bytes(path, canonical_bytes(sample))


def _update_manifest(
    dest: Path,
    samples: list[GeometrySample],
    embedding_files: list[tuple[str, bytes]],
    degradations: list[Degradation],
) -> None:
    path = dest / "manifest.json"
    if not path.is_file():
        return
    manifest = read_model(path, AnalysisManifest)
    previous = sha256_bytes(path.read_bytes())
    frame = manifest.coordinate_frame
    if samples and all(sample.metric_scale == "metric" for sample in samples):
        calibration_id = samples[0].calibration_id
        if calibration_id and all(sample.calibration_id == calibration_id for sample in samples):
            frame = CoordinateFrame(
                name=frame.name, metric_scale="metric", calibration_id=calibration_id
            )
            _rewrite_timeline(dest, frame, degradations)
    artifacts = list(manifest.artifacts)
    if frame != manifest.coordinate_frame:
        artifacts = _refresh_timeline_entry(dest, artifacts)
    known = {entry.path for entry in artifacts}
    for sample in samples:
        millis = int(round(sample.t_sec * 1000))
        rel = f"geometry/{sample.subject_id}/{millis:07d}.json"
        if rel in known:
            continue
        payload = (dest / rel).read_bytes()
        artifacts.append(
            ArtifactEntry(
                path=rel,
                kind="geometry",
                schema_version=SCHEMA_GEOMETRY,
                sha256=sha256_bytes(payload),
                bytes=len(payload),
            )
        )
        known.add(rel)
    for rel, payload in embedding_files:
        if rel in known:
            continue
        artifacts.append(
            ArtifactEntry(
                path=rel,
                kind="embeddings",
                schema_version=SCHEMA_EMBEDDING,
                sha256=sha256_bytes(payload),
                bytes=len(payload),
            )
        )
    stage = StageReport(
        stage="geometry",
        queue_delay_sec=0.0,
        inference_time_sec=0.0,
        processed_frames=len(samples),
        skipped_frames=0,
        trigger_reason="track_observations",
        degradation_flags=[item.code for item in degradations],
    )
    updated = manifest.model_copy(
        update={
            "coordinate_frame": frame,
            "artifacts": artifacts,
            "degradations": [*manifest.degradations, *degradations],
            "stages": [*manifest.stages, stage],
            "supersedes_sha256": previous,
        }
    )
    AnalysisStore(dest).write_manifest(updated)


def _rewrite_timeline(dest: Path, frame: CoordinateFrame, degradations: list[Degradation]) -> None:
    path = dest / "timeline.json"
    timeline = read_model(path, SceneTimeline)
    previous = sha256_bytes(path.read_bytes())
    updated = timeline.model_copy(
        update={
            "coordinate_frame": frame,
            "degradations": [*timeline.degradations, *degradations],
            "supersedes_sha256": previous,
        }
    )
    AnalysisStore(dest).write_timeline(updated)


def _refresh_timeline_entry(dest: Path, artifacts: list[ArtifactEntry]) -> list[ArtifactEntry]:
    payload = (dest / "timeline.json").read_bytes()
    refreshed: list[ArtifactEntry] = []
    for entry in artifacts:
        if entry.path == "timeline.json":
            refreshed.append(
                entry.model_copy(update={"sha256": sha256_bytes(payload), "bytes": len(payload)})
            )
        else:
            refreshed.append(entry)
    return refreshed


def _embedding_file(
    mission_id: str, track_id: str, descriptor: Descriptor, kind: str
) -> tuple[str, bytes]:
    millis = int(round(descriptor.t_sec * 1000))
    name = "prototype.json" if kind == "prototype" else f"{millis:07d}.json"
    rel = f"embeddings/{track_id}/{name}"
    record = TrackEmbedding(
        schema_version=SCHEMA_EMBEDDING,
        mission_id=mission_id,
        track_id=track_id,
        t_sec=descriptor.t_sec,
        kind=kind,  # type: ignore[arg-type]
        model_id=descriptor.space.model_id,
        revision=descriptor.space.revision,
        preprocessing_version=descriptor.space.preprocessing_version,
        dim=descriptor.space.dim,
        vector=[float(value) for value in descriptor.vector],
    )
    return rel, canonical_bytes(record)


def _dedupe_embeddings(files: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    """Keep the last prototype write. Observation files stay in time order."""
    latest: dict[str, bytes] = {}
    order: list[str] = []
    for path, payload in files:
        if path not in latest:
            order.append(path)
        latest[path] = payload
    return [(path, latest[path]) for path in order]


def _describe(appearance: object, view: GeometryView, mask: np.ndarray) -> Descriptor | None:
    try:
        return appearance.describe(view, mask)
    except Exception:
        logger.warning("appearance pooling failed at t=%s", _stamp(view.t_sec))
        return None


def _fields(calibration: object | None, view: GeometryView):
    if calibration is None or getattr(calibration, "failed", False):
        return None
    try:
        return calibration.estimate(view)
    except Exception:
        logger.warning("calibration fallback failed at t=%s", _stamp(view.t_sec))
        return None


def _normals(depth_map: DepthMap, camera: CameraModel) -> np.ndarray:
    normals = (
        depth_map.normals if depth_map.normals is not None else normals_from_depth(depth_map.values)
    )
    return transform_normals(normals, camera)


def _dynamic(rows: list[TrackRecord]) -> bool:
    centers = []
    for record in rows:
        x, y, width, height = record.box.xywh_norm
        centers.append((record.t_sec, x + width * 0.5, y + height * 0.5))
    speeds = []
    for left, right in zip(centers, centers[1:]):
        dt = right[0] - left[0]
        if dt <= 0:
            continue
        speeds.append(math.hypot(right[1] - left[1], right[2] - left[2]) / dt)
    return bool(speeds) and max(speeds) > _MOTION_SPEED


def _mask_from_box(xywh: list[float], width: int, height: int) -> np.ndarray:
    x, y, box_w, box_h = xywh
    cols = (np.arange(width) + 0.5) / max(width, 1)
    rows = (np.arange(height) + 0.5) / max(height, 1)
    grid_x, grid_y = np.meshgrid(cols, rows)
    return (grid_x >= x) & (grid_x < x + box_w) & (grid_y >= y) & (grid_y < y + box_h)


def _frame_name(dest: Path) -> str:
    path = dest / "manifest.json"
    if not path.is_file():
        return "mission_enu"
    return read_model(path, AnalysisManifest).coordinate_frame.name


def _image_size(view: GeometryView, camera: CameraModel | None) -> tuple[int, int]:
    if camera is not None and camera.width > 1 and camera.height > 1:
        return camera.width, camera.height
    image = view.image
    if image is not None and hasattr(image, "size"):
        width, height = image.size
        return int(width), int(height)
    return 64, 64


def _detail(code: str) -> str:
    return {
        "provider_unavailable": "depth provider returned no map; 2D tracks were kept",
        "calibration_missing": "intrinsics or calibration id missing",
        "pose_missing": "camera pose missing; no mission-frame box was written",
        "metric_scale_missing": "metric depth was kept relative because calibration is incomplete",
        "relative_depth_only": "depth has no metric alignment",
        "empty_scene": "no valid depth pixels inside the track",
    }.get(code, code)


def _stamp(t_sec: float) -> float:
    return round(float(t_sec), 6)
