"""Mission-bundle and model-artifact manifests built by today's code match the ss-common contracts.

Bundles are built from `tests/assets/` (plain, and with sidecars from the repository's sidecar
generators plus a DJI srt GPS file); model-artifact manifests come from the FINETUNE handler and
`export_onnx.py` writers. Each manifest must validate against the ODCS-generated model, emit no
field outside the contract, and, where its inputs are fixed, equal the committed golden fixture.
`SS_UPDATE_GOLDEN=1` rewrites those fixtures from the code. The `local-run-nar` and
`onnx-*-nar` fixtures were captured from a real local run; ss-common `make contracts` at tag
`v0.1.0` checks the same snapshots.
"""

import asyncio
import json
import os
import pathlib
import runpy
import shutil
import sys
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

import selfsuvis.scripts.sensors as sensors_pkg
from selfsuvis.pipeline.core.manifests import (
    build_model_artifact,
    finetune_model_artifact,
    model_manifest_path,
    relative_manifest_path,
    utc_timestamp,
)
from selfsuvis.pipeline.fusion.sidecars import load_baro_sidecar, load_imu_sidecar
from selfsuvis.pipeline.fusion.summaries import _build_gps_measurements
from selfsuvis.pipeline.media import mission_bundle
from selfsuvis.pipeline.media.gps import extract_gps
from ss_contracts.models import CONTRACTS

REPO = pathlib.Path(__file__).resolve().parents[3]
ASSETS = REPO / "tests" / "assets"
SENSOR_SCRIPTS = pathlib.Path(sensors_pkg.__file__).resolve().parent
GOLDEN = ASSETS / "contracts" / "golden"
UPDATE = os.environ.get("SS_UPDATE_GOLDEN") == "1"

BUILT_AT = datetime(2026, 9, 19, 8, 0, 0, tzinfo=UTC)
ASSET_VIDEOS = ("vid_testsrc.mp4", "vid_testsrc2.mp4", "vid_green.mp4")

# Two DJI-style fixes; the parser reads the start timecode and the lat/lon/alt text.
DJI_SRT = """1
00:00:00,000 --> 00:00:00,033
<font size="36">FrameCnt : 1, DiffTime : 33ms
[latitude: 50.450100] [longitude: 30.523400] [altitude: 179.000]</font>

2
00:00:01,000 --> 00:00:01,033
<font size="36">FrameCnt : 30, DiffTime : 33ms
[latitude: 50.450300] [longitude: 30.523700] [altitude: 181.500]</font>
"""


def _undeclared(model: BaseModel, payload: Any, path: str = "") -> list[str]:
    """Keys the payload carries that the model (or a nested record) does not declare."""
    if not isinstance(payload, dict):
        return []
    extra = [f"{path}{key}" for key in payload if key not in type(model).model_fields]
    for name in type(model).model_fields:
        value = getattr(model, name)
        nested = payload.get(name)
        if isinstance(value, BaseModel):
            extra += _undeclared(value, nested, f"{path}{name}.")
        elif isinstance(value, list) and isinstance(nested, list):
            for index, (item, raw) in enumerate(zip(value, nested, strict=True)):
                if isinstance(item, BaseModel):
                    extra += _undeclared(item, raw, f"{path}{name}[{index}].")
    return extra


def _validate(contract_id: str, manifest: dict[str, Any]) -> BaseModel:
    model = CONTRACTS[contract_id].model_validate(manifest)
    assert _undeclared(model, manifest) == []
    assert model.model_dump(mode="json", exclude_unset=True) == manifest
    return model


def _check_golden(contract_id: str, name: str, manifest: dict[str, Any]) -> None:
    path = GOLDEN / contract_id / name
    if UPDATE:
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def _generate_sidecars(script: str, stem: str, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = sys.argv
    sys.argv = [script, stem, str(out_dir)]
    try:
        runpy.run_path(str(SENSOR_SCRIPTS / script), run_name="__main__")
    finally:
        sys.argv = argv


@pytest.fixture
def sensor_bundle(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> pathlib.Path:
    """A bundle: one test video, IMU/baro/wind sidecars beside it, env under sensors/, DJI srt."""
    root = tmp_path / "bundle"
    root.mkdir()
    video = root / "vid_testsrc.mp4"
    shutil.copyfile(ASSETS / video.name, video)
    _generate_sidecars("generate_imu_sidecar.py", video.stem, root)
    _generate_sidecars("generate_env_sidecar.py", video.stem, root / "sensors")
    (root / "vid_testsrc.srt").write_text(DJI_SRT, encoding="utf-8")
    capsys.readouterr()
    return root


def _require_ffprobe() -> None:
    if shutil.which("ffprobe") is None:
        pytest.fail(
            "ffprobe is required to probe tests/assets videos; "
            "install ffmpeg (sudo ./scripts/install/install_system_deps.sh)"
        )


def _finetune_handler() -> Any:
    from selfsuvis.worker.handlers import finetune as finetune_handler

    return finetune_handler


@pytest.mark.heavy
def test_manifest_built_from_tests_assets_validates() -> None:
    _require_ffprobe()
    manifest = mission_bundle.build_mission_bundle(
        [ASSETS / name for name in ASSET_VIDEOS], root=ASSETS, created_at=BUILT_AT
    )
    bundle = _validate("mission-bundle", manifest)
    assert [v.video_id for v in bundle.videos] == ["vid_testsrc", "vid_testsrc2", "vid_green"]
    assert all(v.width == 640 and v.height == 360 for v in bundle.videos)
    assert bundle.sidecars == [] and bundle.origin is None
    _check_golden("mission-bundle", "tests-assets.json", manifest)


@pytest.mark.heavy
def test_bundle_with_sidecars_and_gps(sensor_bundle: pathlib.Path) -> None:
    _require_ffprobe()
    video = sensor_bundle / "vid_testsrc.mp4"
    manifest = mission_bundle.build_mission_bundle([video], root=sensor_bundle, created_at=BUILT_AT)
    bundle = _validate("mission-bundle", manifest)
    _check_golden("mission-bundle", "sidecars-and-gps.json", manifest)

    by_kind = {entry.kind: entry for entry in bundle.sidecars}
    assert sorted(by_kind) == ["baro", "env", "gps", "imu", "wind"]
    assert by_kind["env"].path == "sensors/vid_testsrc.env.jsonl"
    assert by_kind["gps"].format == "srt" and by_kind["gps"].row_count == 2
    # Row counts are what the fusion loaders read.
    assert by_kind["imu"].row_count == len(load_imu_sidecar(str(video)))
    assert by_kind["baro"].row_count == len(load_baro_sidecar(str(video)))
    assert (by_kind["imu"].t_start_sec, by_kind["imu"].t_end_sec) == (0.0, 9.995)

    # The origin is the fix GPS extraction gives the first frame and platform fusion anchors on.
    first = extract_gps(str(video), [0.0])[0]
    fusion_origin, _ = _build_gps_measurements([0.0], [first])
    assert bundle.origin is not None
    assert bundle.origin.model_dump() == fusion_origin
    assert bundle.videos[0].gps_source == "srt"


def test_mission_json_is_written_at_the_bundle_root(sensor_bundle: pathlib.Path) -> None:
    manifest = mission_bundle.build_mission_bundle(
        [sensor_bundle / "vid_testsrc.mp4"], root=sensor_bundle, mission_id="m-1", robot_id="uav-7"
    )
    path = mission_bundle.write_mission_bundle(manifest, sensor_bundle)
    assert path == sensor_bundle / "mission.json"
    bundle = _validate("mission-bundle", json.loads(path.read_text(encoding="utf-8")))
    assert (bundle.mission_id, bundle.platform.robot_id) == ("m-1", "uav-7")
    # Rebuilding with mission.json present lists the same files (the manifest is not a sidecar).
    again = mission_bundle.build_mission_bundle(
        [sensor_bundle / "vid_testsrc.mp4"], root=sensor_bundle, mission_id="m-1", robot_id="uav-7"
    )
    assert again["sidecars"] == manifest["sidecars"]


def test_bundle_probe_reads_container_tags(tmp_path: pathlib.Path) -> None:
    probe = {
        "format": {
            "duration": "12.5",
            "tags": {"creation_time": "2026-09-18T12:00:00.000000Z", "model": "FC3170"},
        },
        "streams": [
            {"codec_type": "audio"},
            {"codec_type": "video", "avg_frame_rate": "30000/1001", "width": 3840, "height": 2160},
        ],
    }
    completed = MagicMock(returncode=0, stdout=json.dumps(probe))
    with patch.object(mission_bundle, "run_captured", return_value=completed):
        info = mission_bundle.probe_video(tmp_path / "x.mp4")
    assert info["fps"] == pytest.approx(29.97, abs=1e-2)
    assert (info["duration_sec"], info["width"], info["camera_model"]) == (12.5, 3840, "FC3170")
    assert utc_timestamp(info["creation_time"]) == "2026-09-18T12:00:00Z"


def test_bundle_rejects_files_outside_the_root(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="outside the manifest directory"):
        mission_bundle.build_mission_bundle([ASSETS / "vid_green.mp4"], root=tmp_path)
    with pytest.raises(ValueError, match="at least one video"):
        mission_bundle.build_mission_bundle([], root=tmp_path)
    hidden = tmp_path / ".cache" / "v.mp4"
    hidden.parent.mkdir()
    hidden.write_bytes(b"x")
    with pytest.raises(ValueError, match="starting with a dot"):
        relative_manifest_path(hidden, tmp_path)


def _fake_checkpoint(directory: pathlib.Path, name: str, payload: bytes) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


def test_finetune_manifest_matches_golden(tmp_path: pathlib.Path) -> None:
    ckpt = _fake_checkpoint(tmp_path, "dino_sup_best.pt", b"supervised-finetune-checkpoint\n")
    manifest = finetune_model_artifact(
        ckpt,
        model_version_id="sup_3f2a9c1e",
        base_model="dinov3_vitb14",
        result={"best_accuracy": 0.9125, "distribution_shift": 0.0412, "epochs": 12},
        annotation_count=418,
        created_at=BUILT_AT,
    )
    artifact = _validate("model-artifact", manifest)
    assert artifact.training_data is not None and artifact.training_data.sample_count == 418
    _check_golden("model-artifact", "finetune-checkpoint.json", manifest)


@pytest.mark.heavy
def test_finetune_handler_writes_the_manifest_on_acceptance(tmp_path: pathlib.Path) -> None:
    finetune_handler = _finetune_handler()
    ckpt = _fake_checkpoint(tmp_path, "dino_sup_best.pt", b"weights")
    conn = MagicMock(fetchval=AsyncMock(return_value=37), execute=AsyncMock())
    result = {"best_accuracy": 0.8, "distribution_shift": 0.1, "epochs": 3}
    with (
        patch.object(finetune_handler, "update_job", new_callable=AsyncMock),
        patch.object(finetune_handler.settings, "MODEL_VERSION_ID", "", create=True),
    ):
        version = asyncio.run(
            finetune_handler._persist_finetune_acceptance(
                conn, "abcdef1234", str(ckpt), result, MagicMock(), base_model="dinov3_vitb14"
            )
        )
    manifest = json.loads(model_manifest_path(ckpt).read_text(encoding="utf-8"))
    artifact = _validate("model-artifact", manifest)
    assert (artifact.artifact_id, version) == ("sup_abcdef12", "sup_abcdef12")
    assert artifact.training_data is not None and artifact.training_data.sample_count == 37
    assert {m.name: m.value for m in artifact.metrics}["best_accuracy"] == 0.8


@pytest.mark.heavy
def test_finetune_manifest_failure_only_warns(tmp_path: pathlib.Path) -> None:
    finetune_handler = _finetune_handler()
    logger = MagicMock()
    finetune_handler._write_model_manifest(
        str(tmp_path / "missing.pt"), "sup_1", "dinov3_vitb14", {"best_accuracy": 1.0}, 1,
        BUILT_AT, logger,
    )  # fmt: skip
    logger.warning.assert_called_once()


@pytest.mark.heavy
def test_onnx_export_manifests(tmp_path: pathlib.Path) -> None:
    from ssv_vdp.scripts import export_onnx

    ckpt = _fake_checkpoint(tmp_path / "checkpoints", "dino_ssl_best.pt", b"ssl")
    onnx = _fake_checkpoint(tmp_path / "export", "dino_edge.onnx", b"onnx-fp32")
    int8 = _fake_checkpoint(tmp_path / "export", "dino_edge_int8.onnx", b"onnx-int8")
    args = MagicMock(checkpoint=str(ckpt), image_size=224, opset=18)
    export_onnx._write_export_manifest(str(onnx), args, "dinov2_vitb14_reg", max_diff=5e-6)
    export_onnx._write_export_manifest(
        str(int8), args, "dinov2_vitb14_reg", quantization="int8_static", source=str(onnx)
    )
    fp32 = _validate(
        "model-artifact", json.loads(model_manifest_path(onnx).read_text(encoding="utf-8"))
    )
    assert fp32.derived_from is not None
    assert fp32.derived_from.path == "../checkpoints/dino_ssl_best.pt"
    assert [(m.name, m.value) for m in fp32.metrics] == [("onnx_max_abs_diff", 5e-6)]
    quantized = _validate(
        "model-artifact", json.loads(model_manifest_path(int8).read_text(encoding="utf-8"))
    )
    assert quantized.derived_from is not None and quantized.derived_from.path == "dino_edge.onnx"
    assert (quantized.quantization, quantized.opset, quantized.metrics) == ("int8_static", 18, [])


def test_model_artifact_builder_rules(tmp_path: pathlib.Path) -> None:
    model = _fake_checkpoint(tmp_path, "m.pth", b"x")
    manifest = build_model_artifact(
        model,
        base_model="efficientvit_b1",
        producer="tests",
        metrics={"best_loss": 1.5, "best_recall": float("nan"), "skipped": None},  # type: ignore[dict-item]
    )
    artifact = _validate("model-artifact", manifest)
    assert (artifact.format, artifact.artifact_id) == ("pytorch", "m")
    assert [m.name for m in artifact.metrics] == ["best_loss"]  # non-finite values dropped
    with pytest.raises(ValueError, match="not a model file"):
        build_model_artifact(
            _fake_checkpoint(tmp_path, "m.bin", b"x"), base_model="b", producer="p"
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        build_model_artifact(model, base_model="b", producer="p", created_at=datetime(2026, 1, 1))


def test_real_run_fixtures_are_consistent() -> None:
    """The captured export manifests chain: int8 -> fp32 ONNX -> the run's SSL checkpoint."""
    fp32 = json.loads((GOLDEN / "model-artifact" / "onnx-export-nar.json").read_text())
    int8 = json.loads((GOLDEN / "model-artifact" / "onnx-int8-nar.json").read_text())
    for manifest in (fp32, int8):
        _validate("model-artifact", manifest)
    assert int8["derived_from"] == {"path": fp32["path"], "sha256": fp32["sha256"]}
    assert fp32["derived_from"]["path"] == "../checkpoints/dino_ssl_best.pt"
    _validate(
        "mission-bundle",
        json.loads((GOLDEN / "mission-bundle" / "local-run-nar.json").read_text()),
    )


def test_every_manifest_contract_is_captured() -> None:
    """Contracts without a topic are file manifests, and each has fixtures built above."""
    manifests = {cid for cid, model in CONTRACTS.items() if not model.SS_BINDING.get("mqttTopic")}
    assert manifests == {"mission-bundle", "model-artifact"}
    for contract_id in manifests:
        assert len(list((GOLDEN / contract_id).glob("*.json"))) >= 3
