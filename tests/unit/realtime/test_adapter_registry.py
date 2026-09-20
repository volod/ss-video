from selfsuvis.realtime.adapters import (
    available_occupancy_backends,
    available_pose_backends,
    create_occupancy_adapter,
    create_pose_adapter,
    describe_occupancy_backends,
    describe_pose_backends,
)


def test_pose_adapter_registry_exposes_named_backends():
    adapter = create_pose_adapter("vins_fusion")
    assert adapter.name == "vins_fusion"
    assert "orbslam3" in available_pose_backends()
    assert "liosam" in available_pose_backends()


def test_occupancy_adapter_registry_exposes_named_backends():
    adapter = create_occupancy_adapter("nvblox")
    assert adapter.name == "nvblox"
    assert "voxblox" in available_occupancy_backends()


def test_pose_backend_descriptions_include_selection_metadata():
    backends = describe_pose_backends()
    assert backends["vins_fusion"]["service_name"] == "realtime-vins-fusion"
    assert backends["vins_fusion"]["env_image_var"] == "REALTIME_VINS_FUSION_IMAGE"
    assert backends["vins_fusion"]["open_source"] is True
    assert "camera" in backends["vins_fusion"]["required_modalities"]


def test_occupancy_backend_descriptions_include_docs_and_tradeoffs():
    backends = describe_occupancy_backends()
    assert backends["nvblox"]["hardware_profile"] == "gpu"
    assert backends["nvblox"]["integration_doc"].endswith("nvblox.md")
    assert backends["voxblox"]["cons"]
