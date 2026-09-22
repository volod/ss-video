"""Synthetic reconstruction, scale gating, and the geometry benchmark."""

import math

import numpy as np

from selfsuvis.pipeline.analysis4d.appearance import (
    Descriptor,
    EmbeddingSpace,
    IncompatibleEmbedding,
    PrototypeBank,
    decay_prototype,
)
from selfsuvis.pipeline.analysis4d.camera import (
    PerspectiveEstimate,
    apply_perspective_fields,
    camera_look_at,
    resolve_scale,
)
from selfsuvis.pipeline.analysis4d.geometry_benchmark import (
    _distortion_roundtrip,
    run_geometry_benchmark,
)
from selfsuvis.pipeline.analysis4d.geometry_providers import GeometryView, ScriptedDepth
from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal
from selfsuvis.pipeline.analysis4d.passes import write_profiles
from selfsuvis.pipeline.analysis4d.providers import Prompt, ScriptedGrounding, explain_hub_failure
from selfsuvis.pipeline.analysis4d.reconstruct import (
    fit_gravity_box,
    normals_from_depth,
    transform_normals,
)
from selfsuvis.pipeline.analysis4d.schemas import GeometrySample
from selfsuvis.pipeline.analysis4d.tracks import Detection
from selfsuvis.pipeline.analysis4d.validate import validate_bundle
from selfsuvis.pipeline.workflows.analysis4d_geometry import run_mission_geometry


def test_geometry_benchmark_gates(tmp_path) -> None:
    report = run_geometry_benchmark(output=tmp_path, probe_gpu=False)
    assert report.failures == []
    assert report.passed
    assert report.missing_calibration_scale == "relative"
    assert report.incompatible_embeddings_rejected
    assert report.tracks_preserved_on_provider_failure
    assert report.dynamic_excluded_from_static
    assert report.reprojection_px is not None and report.reprojection_px <= 0.05
    assert report.yaw_error_deg is not None and report.yaw_error_deg <= 1.0


def test_relative_depth_stays_relative() -> None:
    camera = camera_look_at(
        (0.0, -2.0, 1.2),
        (0.0, 0.0, 1.0),
        width=32,
        height=24,
        fx=40.0,
        fy=40.0,
        cx=16.0,
        cy=12.0,
        calibration_id="cal-1",
        metric_alignment=True,
    )
    assert resolve_scale("relative", camera) == "relative"
    assert resolve_scale("metric", camera) == "metric"


def test_perspective_fields_do_not_invent_metric_scale() -> None:
    estimate = PerspectiveEstimate(roll_deg=4.0, pitch_deg=1.0, fov_deg=50.0, confidence=0.6)
    camera = apply_perspective_fields(estimate, width=40, height=30)
    assert camera.fov_deg == 50.0
    assert camera.fx > 0
    assert camera.metric_alignment is False
    assert camera.calibration_id is None
    assert resolve_scale("metric", camera) == "unavailable"


def test_missing_calibration_keeps_depth_boxes_unavailable(tmp_path) -> None:
    mission = "geometry-unscaled"
    dest = tmp_path / "bundle"
    _write_one_track(mission, dest)
    result = run_mission_geometry(
        mission,
        [GeometryView(t_sec=0.0)],
        dest=dest,
        depth=ScriptedDepth(np.full((64, 96), 4.0), "metric"),
    )
    assert result.samples
    assert {sample.metric_scale for sample in result.samples} == {"unavailable"}
    assert all(sample.calibration_id is None for sample in result.samples)
    bundle = validate_bundle(dest)
    assert bundle.manifest.coordinate_frame.metric_scale == "unavailable"
    assert bundle.manifest.coordinate_frame.calibration_id is None


def test_metric_sample_requires_calibration() -> None:
    try:
        GeometrySample(
            schema_version="ss-video.geometry-sample.v1",
            mission_id="mission-a",
            sample_id="geo-1",
            subject_id="track-1",
            t_sec=0.0,
            frame="mission_enu",
            center_m=[0.0, 0.0, 0.0],
            extent_m=[1.0, 1.0, 1.0],
            quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
            metric_scale="metric",
        )
    except ValueError as exc:
        assert "calibration_id" in str(exc)
    else:
        raise AssertionError("metric geometry without calibration_id was accepted")


def test_embedding_decay_and_version_gate() -> None:
    space = EmbeddingSpace("dinov3_vitb14", "rev-1", 2)
    other = EmbeddingSpace("dinov3_vitb14", "rev-2", 2)
    first = Descriptor(space, np.array([1.0, 0.0]), t_sec=0.0)
    nxt = Descriptor(space, np.array([0.0, 1.0]), t_sec=2.0)
    updated = decay_prototype(first, nxt, tau_sec=2.0)
    assert updated.vector[1] > updated.vector[0]
    bank = PrototypeBank(tau_sec=2.0)
    bank.update("track-1", first)
    try:
        bank.update("track-1", Descriptor(other, np.array([0.0, 1.0]), t_sec=1.0))
    except IncompatibleEmbedding:
        return
    raise AssertionError("a new embedding revision updated the prototype")


def test_axis_gate_keeps_a_box_and_drops_a_far_point() -> None:
    rng = np.random.default_rng(0)
    cloud = rng.uniform(-0.4, 0.4, size=(80, 3))
    outlier = np.array([[8.0, 0.0, 0.0]])
    points = np.concatenate([cloud, outlier])
    fitted = fit_gravity_box(points, metric_scale="relative", dynamic=False)
    assert fitted is not None
    assert fitted.extent_m[0] < 2.0
    assert fitted.center_m[0] < 1.0


def _write_one_track(mission: str, dest) -> None:
    write_profiles(
        mission,
        [FrameSignal(t_sec=0.0, embedding=(1.0, 0.0), drift=1.0)],
        [Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")],
        dest,
        grounding=ScriptedGrounding(
            {
                0.0: [
                    Detection(
                        prompt_id="prompt-crate",
                        label_raw="crate",
                        label_normalized="crate",
                        xywh=(0.25, 0.25, 0.40, 0.40),
                        confidence=0.9,
                        appearance=(1.0, 0.0),
                    )
                ]
            }
        ),
        profiles=("fast",),
        created_at="2026-01-15T00:00:00Z",
    )


def test_metric_calibration_is_stored_on_the_mission_frame(tmp_path) -> None:
    mission = "geometry-metric"
    dest = tmp_path / "bundle"
    _write_one_track(mission, dest)
    before = (dest / "tracks.jsonl").read_bytes()
    camera = camera_look_at(
        (0.0, -3.0, 1.4),
        (0.0, 0.0, 1.0),
        width=96,
        height=64,
        fx=80.0,
        fy=80.0,
        cx=48.0,
        cy=32.0,
        calibration_id="cal-synthetic",
        metric_alignment=True,
    )
    result = run_mission_geometry(
        mission,
        [GeometryView(t_sec=0.0, camera=camera)],
        dest=dest,
        depth=ScriptedDepth(np.full((64, 96), 3.0), "metric"),
    )
    after = (dest / "tracks.jsonl").read_bytes()
    assert after == before
    assert result.samples
    assert all(sample.metric_scale == "metric" for sample in result.samples)
    assert all(sample.calibration_id == "cal-synthetic" for sample in result.samples)
    bundle = validate_bundle(dest)
    assert bundle.manifest.coordinate_frame.metric_scale == "metric"
    assert bundle.manifest.coordinate_frame.calibration_id == "cal-synthetic"
    assert bundle.timeline.coordinate_frame == bundle.manifest.coordinate_frame


def test_distortion_roundtrip() -> None:
    assert _distortion_roundtrip() < 1e-3


def test_normals_follow_the_camera_pose() -> None:
    depth = np.full((16, 16), 2.0)
    normals = normals_from_depth(depth)
    assert normals[8, 8, 2] > 0.99
    camera = camera_look_at(
        (0.0, -2.0, 1.2),
        (0.0, 0.0, 1.2),
        width=16,
        height=16,
        fx=20.0,
        fy=20.0,
        cx=8.0,
        cy=8.0,
    )
    world = transform_normals(normals[8:9, 8:9], camera)[0, 0]
    optical = np.asarray(camera.rotation_cam_from_world, dtype=np.float64).T @ np.array(
        [0.0, 0.0, 1.0]
    )
    optical = optical / np.linalg.norm(optical)
    cosine = float(np.dot(world, optical))
    assert cosine > 0.99
    assert abs(math.degrees(math.acos(min(1.0, cosine)))) < 1.0


class _Gated(Exception):
    """Stand-in for a hub 401 on a gated repository."""


def test_gated_hub_failure_names_the_missing_token() -> None:
    message = explain_hub_failure(
        "Intel/zoedepth-nk",
        _Gated("401 Unauthorized hf_notarealtokenvalue123456"),
        token_present=False,
    )
    assert "Intel/zoedepth-nk" in message
    assert "HF_TOKEN is empty" in message
    assert "hf_notarealtokenvalue123456" not in message
    assert "empty depth" not in message


def test_gated_hub_failure_names_a_rejected_token() -> None:
    message = explain_hub_failure(
        "Intel/zoedepth-nyu-kitti",
        OSError(
            "not a valid model identifier listed on 'https://huggingface.co/models'\n"
            "If this is a private repository, make sure to pass a token"
        ),
        token_present=True,
    )
    assert "HF_TOKEN" in message
    assert "not visible" in message
    assert "empty depth" not in message
    assert "pass a token" not in message
    assert "hf_" not in message
