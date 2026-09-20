"""Pose engine adapters."""

from selfsuvis.pipeline.core.docker import REALTIME_ENGINE_IMAGES

from ..config import settings
from .base import RealtimeEngineAdapter, build_descriptor


class StubPoseAdapter(RealtimeEngineAdapter):
    descriptor = build_descriptor(
        name="stub",
        api_url="",
        role="pose",
        provider="selfsuvis",
        open_source=True,
        service_name="realtime-reference",
        hardware_profile="cpu",
        required_modalities=("gps",),
        recommended_modalities=("gps", "imu", "barometer", "magnetometer"),
        pros=(
            "No extra deployment is required.",
            "Useful for local integration tests and API bring-up.",
        ),
        cons=(
            "Not a real SLAM backend.",
            "No loop closure, map optimization, or visual tracking.",
        ),
        integration_doc="docs/runbooks/realtime-reference-sidecar.md",
        notes="Local fused GPS/IMU fallback inside the API process.",
    )


class VinsFusionAdapter(RealtimeEngineAdapter):
    descriptor = build_descriptor(
        name="vins_fusion",
        api_url=settings.vins_fusion_api_url,
        role="pose",
        provider="sidecar",
        service_name="realtime-vins-fusion",
        image=REALTIME_ENGINE_IMAGES["vins_fusion"],
        hardware_profile="cpu_or_gpu",
        required_modalities=("camera", "imu"),
        recommended_modalities=("camera", "imu", "gps"),
        pros=(
            "Strong visual-inertial pose estimation on drone-class RGB + IMU feeds.",
            "Well-known open-source baseline for tightly coupled VIO.",
        ),
        cons=(
            "Needs reliable camera/IMU calibration.",
            "Can drift in weak-texture scenes without GPS priors or relocalization support.",
        ),
        integration_doc="docs/runbooks/realtime-sidecars/vins-fusion.md",
        notes="RGB + IMU + GPS visual-inertial fusion sidecar.",
    )


class OrbSlam3Adapter(RealtimeEngineAdapter):
    descriptor = build_descriptor(
        name="orbslam3",
        api_url=settings.orbslam3_api_url,
        role="pose",
        provider="sidecar",
        service_name="realtime-orbslam3",
        image=REALTIME_ENGINE_IMAGES["orbslam3"],
        hardware_profile="cpu",
        required_modalities=("camera",),
        recommended_modalities=("camera", "imu"),
        pros=(
            "Good fallback when LiDAR is unavailable and monocular/stereo SLAM is enough.",
            "Supports several camera modes and has broad community familiarity.",
        ),
        cons=(
            "Can be fragile under aggressive motion blur and repeated low-texture patterns.",
            "Operational tuning and vocabulary management are heavier than the stub path.",
        ),
        integration_doc="docs/runbooks/realtime-sidecars/orbslam3.md",
        notes="Fallback visual SLAM sidecar for RGB(+IMU) feeds.",
    )


class LioSamAdapter(RealtimeEngineAdapter):
    descriptor = build_descriptor(
        name="liosam",
        api_url=settings.liosam_api_url,
        role="pose",
        provider="sidecar",
        service_name="realtime-liosam",
        image=REALTIME_ENGINE_IMAGES["liosam"],
        hardware_profile="cpu",
        required_modalities=("lidar", "imu"),
        recommended_modalities=("lidar", "imu", "gps"),
        pros=(
            "Best fit when the drone carries LiDAR and needs robust pose through low-texture scenes.",
            "Loop-closure-friendly mapping path for longer flights.",
        ),
        cons=(
            "Requires LiDAR hardware and tighter time synchronization discipline.",
            "Heavier integration surface than pure visual estimators.",
        ),
        integration_doc="docs/runbooks/realtime-sidecars/lio-sam.md",
        notes="RGB + IMU + LiDAR sidecar deployment.",
    )
