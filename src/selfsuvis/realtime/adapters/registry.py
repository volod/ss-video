"""Adapter registry for realtime engines."""

from .base import (
    RealtimeEngineAdapter,
    available_backend_urls,
    describe_backends,
    instantiate_adapter,
)
from .occupancy import NvbloxAdapter, StubOccupancyAdapter, VoxbloxAdapter
from .pose import LioSamAdapter, OrbSlam3Adapter, StubPoseAdapter, VinsFusionAdapter

_POSE: dict[str, type[RealtimeEngineAdapter]] = {
    "stub": StubPoseAdapter,
    "vins_fusion": VinsFusionAdapter,
    "orbslam3": OrbSlam3Adapter,
    "liosam": LioSamAdapter,
}

_OCCUPANCY: dict[str, type[RealtimeEngineAdapter]] = {
    "stub": StubOccupancyAdapter,
    "nvblox": NvbloxAdapter,
    "voxblox": VoxbloxAdapter,
}


def create_pose_adapter(name: str) -> RealtimeEngineAdapter:
    return instantiate_adapter(_POSE, name)


def create_occupancy_adapter(name: str) -> RealtimeEngineAdapter:
    return instantiate_adapter(_OCCUPANCY, name)


def available_pose_backends() -> dict[str, str]:
    return available_backend_urls(_POSE)


def available_occupancy_backends() -> dict[str, str]:
    return available_backend_urls(_OCCUPANCY)


def describe_pose_backends() -> dict[str, dict[str, object]]:
    return describe_backends(_POSE)


def describe_occupancy_backends() -> dict[str, dict[str, object]]:
    return describe_backends(_OCCUPANCY)
