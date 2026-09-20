"""Occupancy engine adapters."""

from selfsuvis.pipeline.core.docker import REALTIME_ENGINE_IMAGES

from ..config import settings
from .base import RealtimeEngineAdapter, build_descriptor


class StubOccupancyAdapter(RealtimeEngineAdapter):
    descriptor = build_descriptor(
        name="stub",
        api_url="",
        role="occupancy",
        provider="selfsuvis",
        open_source=True,
        service_name="realtime-reference",
        hardware_profile="cpu",
        required_modalities=("pose",),
        recommended_modalities=("pose", "depth"),
        pros=(
            "Runs everywhere and writes inspectable JSON tiles.",
            "Good for validating the API contract before deploying a real mapper.",
        ),
        cons=(
            "Not a volumetric mapper.",
            "No TSDF / ESDF / occupancy fusion quality guarantees.",
        ),
        integration_doc="docs/runbooks/realtime-reference-sidecar.md",
        notes="Local stub tile writer under .data/maps/realtime/.",
    )


class NvbloxAdapter(RealtimeEngineAdapter):
    descriptor = build_descriptor(
        name="nvblox",
        api_url=settings.nvblox_api_url,
        role="occupancy",
        provider="sidecar",
        service_name="realtime-nvblox",
        image=REALTIME_ENGINE_IMAGES["nvblox"],
        hardware_profile="gpu",
        required_modalities=("pose", "depth"),
        recommended_modalities=("pose", "depth", "camera"),
        pros=(
            "Strong choice for GPU-backed dense TSDF / ESDF mapping.",
            "Well suited to obstacle-aware planning stacks that need signed distance fields.",
        ),
        cons=(
            "GPU requirement makes it heavier to deploy on small edge boxes.",
            "Needs stable depth and pose to avoid map tearing.",
        ),
        integration_doc="docs/runbooks/realtime-sidecars/nvblox.md",
        notes="GPU occupancy mapping sidecar.",
    )


class VoxbloxAdapter(RealtimeEngineAdapter):
    descriptor = build_descriptor(
        name="voxblox",
        api_url=settings.voxblox_api_url,
        role="occupancy",
        provider="sidecar",
        service_name="realtime-voxblox",
        image=REALTIME_ENGINE_IMAGES["voxblox"],
        hardware_profile="cpu",
        required_modalities=("pose", "depth"),
        recommended_modalities=("pose", "depth", "camera"),
        pros=(
            "CPU-friendly volumetric mapping option.",
            "Good fit when the deployment target has no CUDA-capable GPU.",
        ),
        cons=(
            "Lower throughput than nvblox on dense depth streams.",
            "Needs careful resolution tuning to stay real-time on weak CPUs.",
        ),
        integration_doc="docs/runbooks/realtime-sidecars/voxblox.md",
        notes="CPU occupancy mapping sidecar.",
    )
