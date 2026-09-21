"""Geometry benchmark for back-projection, scale, and appearance versions.

``python -m selfsuvis.pipeline.analysis4d.geometry_benchmark`` checks the pinned
synthetic camera, refuses metric scale without calibration, rejects mismatched
embedding versions, and keeps 2D tracks when depth fails. ``--skip-gpu`` omits
the depth and DINO probes.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from selfsuvis.pipeline.analysis4d.appearance import (
    Descriptor,
    EmbeddingSpace,
    IncompatibleEmbedding,
    cosine_association,
)
from selfsuvis.pipeline.analysis4d.camera import (
    PerspectiveEstimate,
    apply_perspective_fields,
    camera_from_mapping_pose,
    camera_look_at,
    mapping_pose_dict,
    project,
    resolve_scale,
    unproject,
)
from selfsuvis.pipeline.analysis4d.geometry_providers import (
    FailingDepth,
    GeometryView,
    ScriptedDepth,
)
from selfsuvis.pipeline.analysis4d.io import sha256_bytes, write_bytes
from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal
from selfsuvis.pipeline.analysis4d.passes import write_profiles
from selfsuvis.pipeline.analysis4d.pin import ProviderProbe, meets_gate
from selfsuvis.pipeline.analysis4d.providers import Prompt, ScriptedGrounding
from selfsuvis.pipeline.analysis4d.reconstruct import (
    StaticCloud,
    aabb_at_yaw,
    fit_gravity_box,
    reprojection_error_px,
    yaw_error_deg,
)
from selfsuvis.pipeline.analysis4d.synthetic import load_synthetic_scene, render_scene
from selfsuvis.pipeline.analysis4d.tracks import Detection
from selfsuvis.pipeline.analysis4d.validate import validate_bundle
from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.workflows.analysis4d_geometry import run_mission_geometry

logger = get_logger(__name__)

SCHEMA_GEOMETRY_BENCHMARK = "ss-video.geometry-benchmark.v1"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeometryBenchmarkReport(_Model):
    """Machine-readable result of the geometry benchmark."""

    schema_version: str = SCHEMA_GEOMETRY_BENCHMARK
    passed: bool
    reprojection_px: float | None = None
    box_center_error_m: float | None = None
    box_extent_error_m: float | None = None
    yaw_error_deg: float | None = None
    solid_center_bias_m: float | None = None
    missing_calibration_scale: str = ""
    incompatible_embeddings_rejected: bool = False
    tracks_preserved_on_provider_failure: bool = False
    dynamic_excluded_from_static: bool = False
    probes: list[ProviderProbe] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


def default_scene_path() -> Path:
    """Return the pinned synthetic-camera fixture."""
    from selfsuvis.pipeline.core import settings

    return Path(settings.PROJECT_ROOT) / "tests" / "assets" / "synthetic-camera" / "scene.json"


def default_geometry_report_path() -> Path:
    """Return ``$DATA_DIR/analysis/_benchmark/geometry-report.json``."""
    from selfsuvis.pipeline.analysis4d.paths import benchmark_report_path

    return benchmark_report_path().with_name("geometry-report.json")


def run_geometry_benchmark(
    scene: Path | None = None,
    output: Path | None = None,
    *,
    probe_gpu: bool = False,
) -> GeometryBenchmarkReport:
    """Score the synthetic camera and the scale, embedding, and failure gates.

    Args:
        scene: Synthetic camera JSON. Defaults to the pinned fixture.
        output: Directory for the track bundles used by the failure and motion gates.
        probe_gpu: When true, load depth and DINO candidates on CUDA.

    Returns:
        The report. ``passed`` is false when any gate fails.
    """
    fixture = load_synthetic_scene(Path(scene) if scene is not None else default_scene_path())
    failures: list[str] = []
    rendered = render_scene(fixture)
    z = rendered.depth_m[rendered.mask]
    recovered = unproject(rendered.pixels, z, rendered.camera)
    reprojection = reprojection_error_px(rendered.pixels, recovered, rendered.camera)
    if reprojection > fixture.reprojection_px_max:
        failures.append(f"reprojection_px {reprojection}")
    pose = mapping_pose_dict(rendered.camera)
    restored = camera_from_mapping_pose(
        pose,
        width=rendered.camera.width,
        height=rendered.camera.height,
        fx=rendered.camera.fx,
        fy=rendered.camera.fy,
        cx=rendered.camera.cx,
        cy=rendered.camera.cy,
        calibration_id=rendered.camera.calibration_id,
        metric_alignment=True,
    )
    mapped = unproject(rendered.pixels, z, restored)
    if float(np.linalg.norm(mapped - rendered.points_world, axis=1).mean()) > 1e-4:
        failures.append("ss-mapping pose dict did not round-trip the synthetic points")
    fitted = fit_gravity_box(rendered.points_world, metric_scale="metric", dynamic=False)
    center_error = extent_error = yaw_error = solid_bias = None
    if fitted is None:
        failures.append("gravity box fit returned no box")
    else:
        center_ref, extent_ref = aabb_at_yaw(
            rendered.points_world, math.radians(fixture.box_yaw_deg)
        )
        center_error = float(np.linalg.norm(np.array(fitted.center_m) - center_ref))
        extent_error = _extent_error(
            fitted.extent_m, extent_ref, fitted.yaw_rad, fixture.box_yaw_deg
        )
        yaw_error = yaw_error_deg(fitted.yaw_rad, fixture.box_yaw_deg)
        solid_bias = float(
            np.linalg.norm(np.array(fitted.center_m) - np.array(fixture.box_center_m))
        )
        if center_error > fixture.box_center_error_m_max:
            failures.append(f"box_center_error_m {center_error}")
        if extent_error > fixture.box_extent_error_m_max:
            failures.append(f"box_extent_error_m {extent_error}")
        if yaw_error > fixture.yaw_error_deg_max:
            failures.append(f"yaw_error_deg {yaw_error}")
        if solid_bias > fixture.solid_center_bias_m_max:
            failures.append(f"solid_center_bias_m {solid_bias}")
    missing = _missing_calibration(rendered.camera)
    if missing == "metric":
        failures.append("missing calibration produced metric scale")
    rejected = _reject_embeddings()
    if not rejected:
        failures.append("incompatible embedding versions were associated")
    preserved = False
    dynamic_ok = False
    if output is not None:
        root = Path(output)
        if root.exists():
            shutil.rmtree(root)
        preserved = _provider_failure_keeps_tracks(root / "failure")
        dynamic_ok = _dynamic_stays_out_of_static(root / "dynamic")
        if not preserved:
            failures.append("provider failure changed the 2D tracks")
        if not dynamic_ok:
            failures.append("dynamic points were inserted into the static cloud")
    else:
        failures.append("geometry benchmark requires an output directory")
    probes: list[ProviderProbe] = []
    if probe_gpu:
        probes = _probe(failures)
    return GeometryBenchmarkReport(
        passed=not failures,
        reprojection_px=reprojection,
        box_center_error_m=center_error,
        box_extent_error_m=extent_error,
        yaw_error_deg=yaw_error,
        solid_center_bias_m=solid_bias,
        missing_calibration_scale=missing,
        incompatible_embeddings_rejected=rejected,
        tracks_preserved_on_provider_failure=preserved,
        dynamic_excluded_from_static=dynamic_ok,
        probes=probes,
        failures=failures,
    )


def _extent_error(
    fitted: tuple[float, float, float],
    reference: np.ndarray,
    fitted_yaw: float,
    truth_deg: float,
) -> float:
    raw = abs(math.degrees(fitted_yaw) - truth_deg) % 180.0
    raw = min(raw, 180.0 - raw)
    extent = np.array(fitted, dtype=np.float64)
    if raw > 45.0:
        extent = np.array([extent[1], extent[0], extent[2]])
    return float(np.max(np.abs(extent - reference)))


def _missing_calibration(camera) -> str:
    """Metric depth without a calibration id or a pose must not stay metric."""
    bare = camera_look_at(
        (-2.2, -2.6, 1.5),
        (0.0, 0.0, 0.7),
        width=camera.width,
        height=camera.height,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        calibration_id=None,
        metric_alignment=False,
    )
    no_pose = apply_perspective_fields(
        PerspectiveEstimate(roll_deg=3.0, pitch_deg=-2.0, fov_deg=55.0, confidence=0.4),
        width=camera.width,
        height=camera.height,
    )
    relative = resolve_scale("relative", camera)
    downgraded = resolve_scale("metric", bare)
    unavailable = resolve_scale("metric", no_pose)
    if relative != "relative" or downgraded != "relative":
        return "metric"
    if unavailable == "metric":
        return "metric"
    return downgraded if unavailable == "unavailable" else unavailable


def _reject_embeddings() -> bool:
    left = EmbeddingSpace(model_id="dinov3_vitb14", revision="sha256:" + "ab" * 32, dim=4)
    right = EmbeddingSpace(model_id="dinov3_vitb14", revision="sha256:" + "cd" * 32, dim=4)
    vector = np.array([1.0, 0.0, 0.0, 0.0])
    try:
        cosine_association(Descriptor(left, vector), Descriptor(right, vector))
    except IncompatibleEmbedding:
        same = cosine_association(Descriptor(left, vector), Descriptor(left, vector))
        return same > 0.99
    return False


def _provider_failure_keeps_tracks(dest: Path) -> bool:
    mission = "geometry-failure"
    _write_track_bundle(
        dest, mission, {0.0: (0.30, 0.30, 0.20, 0.25), 1.0: (0.32, 0.30, 0.20, 0.25)}
    )
    before = (dest / "tracks.jsonl").read_bytes()
    view = _view(0.0)
    result = run_mission_geometry(
        mission,
        [view, _view(1.0)],
        dest=dest,
        depth=FailingDepth(),
    )
    after = (dest / "tracks.jsonl").read_bytes()
    if after != before or sha256_bytes(after) != result.tracks_sha256:
        return False
    if result.samples:
        return False
    if "provider_unavailable" not in result.degradations:
        return False
    validate_bundle(dest)
    return True


def _dynamic_stays_out_of_static(dest: Path) -> bool:
    mission = "geometry-dynamic"
    _write_track_bundle(
        dest, mission, {0.0: (0.15, 0.25, 0.40, 0.40), 1.0: (0.35, 0.25, 0.40, 0.40)}
    )
    cloud = StaticCloud()
    depth = np.full((64, 96), 3.0)
    views = [_view(0.0, depth_shape=depth.shape), _view(1.0, depth_shape=depth.shape)]
    result = run_mission_geometry(
        mission,
        views,
        dest=dest,
        depth=ScriptedDepth(depth, "metric"),
        static_cloud=cloud,
    )
    validate_bundle(dest)
    if not result.samples or any(not sample.dynamic for sample in result.samples):
        return False
    if any(sample.metric_scale == "metric" for sample in result.samples):
        return False
    return result.static_ids == [] and not cloud.points


def _view(t_sec: float, depth_shape: tuple[int, int] | None = None) -> GeometryView:
    camera = camera_look_at(
        (0.0, -3.0, 1.4),
        (0.0, 0.0, 1.0),
        width=96,
        height=64,
        fx=80.0,
        fy=80.0,
        cx=48.0,
        cy=32.0,
        calibration_id=None,
        metric_alignment=False,
    )
    mask = None
    if depth_shape is not None:
        mask = np.ones(depth_shape, dtype=bool)
        mask[:4, :] = False
    return GeometryView(t_sec=t_sec, camera=camera, mask=mask)


def _write_track_bundle(
    dest: Path, mission: str, boxes: dict[float, tuple[float, float, float, float]]
) -> None:
    frames = []
    table: dict[float, list[Detection]] = {}
    for index, (stamp, xywh) in enumerate(sorted(boxes.items())):
        embedding = (1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)
        frames.append(FrameSignal(t_sec=stamp, embedding=embedding, drift=1.0))
        table[stamp] = [
            Detection(
                prompt_id="prompt-crate",
                label_raw="crate",
                label_normalized="crate",
                xywh=xywh,
                confidence=0.9,
                appearance=(1.0, 0.0),
            )
        ]
    write_profiles(
        mission,
        frames,
        [Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")],
        dest,
        grounding=ScriptedGrounding(table),
        profiles=("fast",),
        created_at="2026-01-15T00:00:00Z",
    )


def _probe(failures: list[str]) -> list[ProviderProbe]:
    from selfsuvis.pipeline.analysis4d.geometry_providers import probe_geometry_models
    from selfsuvis.pipeline.analysis4d.pin import (
        PINNED_APPEARANCE,
        PINNED_APPEARANCE_REVISION,
        PINNED_METRIC_DEPTH,
        PINNED_METRIC_REVISION,
        PINNED_METRIC_WEIGHTS,
        PINNED_RELATIVE_DEPTH,
        PINNED_RELATIVE_REVISION,
        PINNED_RELATIVE_WEIGHTS,
    )

    logger.info("Probing 4D geometry providers")
    probes = probe_geometry_models()
    by_id = {item.provider_id: item for item in probes}
    relative = by_id.get("depth_anything")
    if relative is None or not relative.available or not meets_gate(relative):
        failures.append("relative depth provider missed the license, memory, or latency gate")
    elif PINNED_RELATIVE_REVISION and PINNED_RELATIVE_WEIGHTS:
        if PINNED_RELATIVE_DEPTH not in relative.detail:
            failures.append("relative depth model does not match the pin")
        if (
            PINNED_RELATIVE_REVISION not in relative.detail
            or PINNED_RELATIVE_WEIGHTS not in relative.detail
        ):
            failures.append(f"relative depth cache {relative.detail} does not match the pin")
    metric = by_id.get("depth_anything_metric")
    if metric is None or not metric.available or not meets_gate(metric):
        failures.append("metric depth provider missed the license, memory, or latency gate")
    elif PINNED_METRIC_REVISION and PINNED_METRIC_WEIGHTS:
        if PINNED_METRIC_DEPTH not in metric.detail:
            failures.append("metric depth model does not match the pin")
        if (
            PINNED_METRIC_REVISION not in metric.detail
            or PINNED_METRIC_WEIGHTS not in metric.detail
        ):
            failures.append(f"metric depth cache {metric.detail} does not match the pin")
    zoedepth = by_id.get("zoedepth")
    if zoedepth is not None and "HF_TOKEN" in (zoedepth.detail or ""):
        failures.append(zoedepth.detail)
    appearance = by_id.get("dinov3")
    if appearance is None or not appearance.available or not meets_gate(appearance):
        failures.append("masked DINO provider missed the license, memory, or latency gate")
    elif PINNED_APPEARANCE_REVISION and PINNED_APPEARANCE not in (appearance.detail or ""):
        failures.append("appearance model does not match the pin")
    if (
        appearance is not None
        and PINNED_APPEARANCE_REVISION
        and PINNED_APPEARANCE_REVISION not in appearance.detail
    ):
        failures.append(f"appearance revision {appearance.detail} does not match the pin")
    return probes


def _distortion_roundtrip() -> float:
    camera = camera_look_at(
        (0.0, -2.0, 1.2),
        (0.0, 0.0, 1.0),
        width=96,
        height=64,
        fx=70.0,
        fy=70.0,
        cx=48.0,
        cy=32.0,
        k1=-0.12,
        k2=0.02,
    )
    points = np.array([[0.2, 0.4, 1.1], [-0.3, 0.2, 0.8], [0.0, 0.0, 1.4]], dtype=np.float64)
    pixels, depth = project(points, camera)
    recovered = unproject(pixels, depth, camera)
    return float(np.linalg.norm(recovered - points, axis=1).max())


def main(argv: list[str] | None = None) -> int:
    """Write the geometry benchmark report. Returns 0 when it passes."""
    parser = argparse.ArgumentParser(description="4D geometry benchmark")
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args(argv)
    report_path = args.output or default_geometry_report_path()
    bundle_dir = args.bundle_dir or report_path.parent / "geometry"
    report = run_geometry_benchmark(
        args.scene,
        bundle_dir,
        probe_gpu=not args.skip_gpu,
    )
    distortion = _distortion_roundtrip()
    if distortion > 1e-3:
        report.failures.append(f"distortion_roundtrip_m {distortion}")
        report.passed = not report.failures
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    write_bytes(report_path, payload.encode("utf-8"))
    logger.info("geometry benchmark passed=%s report=%s", report.passed, report_path)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
