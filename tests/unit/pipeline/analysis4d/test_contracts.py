"""Schema round-trip, rejection fixtures, geometry, and the append-only store."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from selfsuvis.pipeline.analysis4d.corpus import build_corpus
from selfsuvis.pipeline.analysis4d.geometry import (
    contains,
    metric_series_feasible,
    project_pinhole,
    relation_holds,
    reprojection_residual_px,
    velocity_feasible,
)
from selfsuvis.pipeline.analysis4d.io import sha256_bytes
from selfsuvis.pipeline.analysis4d.schemas import SceneTimeline
from selfsuvis.pipeline.analysis4d.store import AnalysisStore, AppendOnlyError
from selfsuvis.pipeline.analysis4d.validate import (
    ContractError,
    issues_from_validation,
    validate_bundle,
)
from selfsuvis.worker.artifacts import analysis_artifact_dir

REPO = Path(__file__).resolve().parents[4]
ASSETS = REPO / "tests" / "assets" / "analysis4d"

SPEC_TIMELINE = {
    "schema_version": "ss-video.scene-timeline.v1",
    "mission_id": "mission-42",
    "profile": "fast",
    "coordinate_frame": {
        "name": "mission_enu",
        "metric_scale": "metric",
        "calibration_id": "cal-7",
    },
    "model_manifest_ref": "manifest.json",
    "degradations": [],
    "events": [
        {
            "event_id": "evt-19",
            "type": "entered_region",
            "summary": "track-8 entered loading-zone",
            "start_sec": 12.4,
            "end_sec": 14.1,
            "participants": ["track-8", "region-loading-zone"],
            "state_delta_refs": ["delta-103", "delta-104"],
            "location": {
                "frame": "mission_enu",
                "center_m": [4.2, -1.1, 0.7],
                "covariance_diag": [0.08, 0.09, 0.16],
            },
            "confidence": 0.94,
            "verification": {
                "status": "accepted",
                "rules": ["lifetime_overlap", "obb_region_intersection", "reprojection"],
                "vlm_claim_ref": "proposal-88",
                "reasons": [],
            },
            "evidence": [
                {
                    "frame_id": "mission-42:17:12400",
                    "t_sec": 12.4,
                    "track_ids": ["track-8"],
                    "mask_ref": "masks/track-8/12400.rle",
                    "geometry_ref": "geometry/track-8/12400.json",
                }
            ],
            "supersedes": None,
        }
    ],
    "qa_pairs": [
        {
            "qa_id": "qa-31",
            "type": "spatial",
            "question": "Which tracked object entered the loading zone after 12 seconds?",
            "answer": {"kind": "track_ref", "value": "track-8", "unit": None},
            "graph_program": "entered(?track, region-loading-zone, after=12.0)",
            "interval_sec": [12.0, 15.0],
            "evidence_event_ids": ["evt-19"],
            "verification_status": "accepted",
        }
    ],
}


def _issues(document: dict) -> list[str]:
    with pytest.raises(ValidationError) as caught:
        SceneTimeline.model_validate(document)
    return issues_from_validation(caught.value)


def test_spec_timeline_round_trip() -> None:
    model = SceneTimeline.model_validate(SPEC_TIMELINE)
    dumped = model.model_dump(mode="json", exclude_unset=True)
    assert dumped == SPEC_TIMELINE
    assert SceneTimeline.model_validate(dumped) == model


def test_unversioned_interval_and_mixed_frame_fixtures() -> None:
    invalid = ASSETS / "invalid"
    unversioned = json.loads((invalid / "unversioned.json").read_text(encoding="utf-8"))
    interval = json.loads((invalid / "invalid-interval.json").read_text(encoding="utf-8"))
    mixed = json.loads((invalid / "mixed-frame.json").read_text(encoding="utf-8"))
    assert any(item.startswith("unversioned:") for item in _issues(unversioned))
    assert any(item.startswith("invalid_interval:") for item in _issues(interval))
    assert any("mixed_coordinate_frame" in item for item in _issues(mixed))


def test_unknown_field_fails_closed() -> None:
    document = dict(SPEC_TIMELINE)
    document["extra"] = 1
    assert any(item.startswith("unknown_field:") for item in _issues(document))


def test_dangling_id_and_geometry_contradiction() -> None:
    with pytest.raises(ContractError) as dangling:
        validate_bundle(ASSETS / "invalid" / "dangling")
    assert any(item.startswith("dangling_id:") for item in dangling.value.issues)
    with pytest.raises(ContractError) as contradicted:
        validate_bundle(ASSETS / "invalid" / "bad-edge")
    assert any("contradicts geometry" in item for item in contradicted.value.issues)


def test_nominal_bundle_validates() -> None:
    bundle = validate_bundle(ASSETS / "nominal", require_truth=True)
    assert bundle.timeline.events[0].verification.status == "accepted"
    assert bundle.manifest.coordinate_frame.metric_scale == "metric"


def test_geometry_predicates_and_reprojection() -> None:
    assert contains([0, 0, 0], [4, 4, 4], [0, 0, 0], [1, 1, 1])
    assert not contains([0, 0, 0], [1, 1, 1], [2, 0, 0], [1, 1, 1])
    assert relation_holds("left_of", [0, 0, 0], [1, 1, 1], [2, 0, 0], [1, 1, 1])
    assert not relation_holds("right_of", [0, 0, 0], [1, 1, 1], [2, 0, 0], [1, 1, 1])
    assert project_pinhole([0, 0, 10], focal_px=100, principal_px=(50, 50)) == (50, 50)
    assert reprojection_residual_px([0, 0, 10], (50, 50), focal_px=100, principal_px=(50, 50)) == 0
    assert velocity_feasible([(0.0, [0, 0, 0]), (1.0, [10, 0, 0])])
    assert not velocity_feasible([(0.0, [0, 0, 0]), (1.0, [100, 0, 0])])
    jump = [
        ("unavailable", 0.0, [0.0, 0.0, 1.0]),
        ("unavailable", 1.0, [0.0, 0.0, 3000.0]),
    ]
    assert metric_series_feasible(jump)
    assert not metric_series_feasible(
        [("metric", 0.0, [0.0, 0.0, 0.0]), ("metric", 1.0, [100.0, 0.0, 0.0])]
    )
    assert metric_series_feasible(
        [("metric", 0.0, [0.0, 0.0, 0.0]), ("metric", 1.0, [10.0, 0.0, 0.0])]
    )


def test_store_is_append_only_and_records_supersession(tmp_path: Path) -> None:
    source = validate_bundle(ASSETS / "empty-scene")
    store = AnalysisStore(tmp_path)
    store.write_timeline(source.timeline)
    previous = sha256_bytes((tmp_path / "timeline.json").read_bytes())
    replacement = source.timeline.model_copy(
        update={"supersedes_sha256": previous, "profile": "deep"}
    )
    store.write_timeline(replacement)
    assert list((tmp_path / "history").glob("timeline-*.json"))
    with pytest.raises(AppendOnlyError):
        store.write_timeline(replacement.model_copy(update={"profile": "fast"}))
    track = source.tracks
    assert track == []
    from selfsuvis.pipeline.analysis4d.corpus import _box, _track

    observation = _track(
        "mission-empty", "obs-1", "track-1", "confirmed", 0.0, _box(0.1, 0.1, 0.2, 0.2)
    )
    store.append_track(observation)
    with pytest.raises(AppendOnlyError):
        store.append_track(observation)


def test_worker_artifact_dir_stays_under_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert analysis_artifact_dir("mission-1") == tmp_path / "analysis" / "mission-1" / "4d"
    with pytest.raises(ValueError):
        analysis_artifact_dir("../mission")


def test_builder_matches_committed_assets(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    built = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    committed = {
        path.relative_to(ASSETS).as_posix(): path.read_bytes()
        for path in ASSETS.rglob("*")
        if path.is_file()
    }
    assert built.keys() == committed.keys()
    assert built == committed
