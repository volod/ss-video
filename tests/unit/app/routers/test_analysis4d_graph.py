"""Read routes for the temporal scene graph."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from selfsuvis.app.deps import rate_limit, require_api_key
from selfsuvis.app.routers.analysis4d import router
from selfsuvis.pipeline.analysis4d.graph import RegionBox
from selfsuvis.pipeline.analysis4d.graph_benchmark import prepare_pinned_scene
from selfsuvis.pipeline.analysis4d.vlm import UnavailableVlm
from selfsuvis.pipeline.workflows.analysis4d_graph import run_mission_graph


def test_scene_graph_routes_return_deltas_and_filter_time(tmp_path, monkeypatch) -> None:
    prepare_pinned_scene(tmp_path)
    run_mission_graph(
        "mission-graph",
        dest=tmp_path,
        provider=UnavailableVlm(),
        regions=[RegionBox("region-1", "loading-zone", [0.0, 0.0, 0.0], [10.0, 10.0, 4.0])],
    )

    def _dir(mission_id: str):
        if mission_id != "mission-graph":
            raise ValueError("mission_id must be a single path segment")
        return tmp_path

    monkeypatch.setattr("selfsuvis.app.routers.analysis4d.analysis_dir", _dir)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[rate_limit] = lambda: None
    client = TestClient(app)

    graph = client.get("/analysis/mission-graph/4d/graph")
    assert graph.status_code == 200
    body = graph.json()
    assert body["nodes"]
    assert body["edges"]
    assert any(edge["predicate"] == "contains" for edge in body["edges"])

    during = client.get("/analysis/mission-graph/4d/graph", params={"t_sec": 1.0})
    assert during.status_code == 200
    assert during.json()["edges"]
    after = client.get("/analysis/mission-graph/4d/graph", params={"t_sec": 100.0})
    assert after.status_code == 200
    assert after.json()["edges"] == []

    deltas = client.get("/analysis/mission-graph/4d/deltas")
    assert deltas.status_code == 200
    assert deltas.json()["deltas"]

    proposals = client.get("/analysis/mission-graph/4d/proposals")
    assert proposals.status_code == 200
    assert proposals.json()["proposals"] == []

    missing = client.get("/analysis/other-mission/4d/graph")
    assert missing.status_code == 404


def test_timeline_and_qa_routes_return_evidence(tmp_path, monkeypatch) -> None:
    from selfsuvis.pipeline.analysis4d.review import UnavailableReview
    from selfsuvis.pipeline.workflows.analysis4d_verify import run_mission_verify

    prepare_pinned_scene(tmp_path)
    run_mission_graph(
        "mission-graph",
        dest=tmp_path,
        provider=UnavailableVlm(),
        regions=[RegionBox("region-1", "loading-zone", [0.0, 0.0, 0.0], [10.0, 10.0, 4.0])],
    )
    run_mission_verify("mission-graph", dest=tmp_path, reviewer=UnavailableReview())

    def _dir(mission_id: str):
        if mission_id != "mission-graph":
            raise ValueError("mission_id must be a single path segment")
        return tmp_path

    monkeypatch.setattr("selfsuvis.app.routers.analysis4d.analysis_dir", _dir)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[rate_limit] = lambda: None
    client = TestClient(app)

    timeline = client.get("/analysis/mission-graph/4d/timeline")
    assert timeline.status_code == 200
    body = timeline.json()
    assert body["schema_version"] == "ss-video.scene-timeline.v1"
    accepted = [event for event in body["events"] if event["verification"]["status"] == "accepted"]
    assert accepted
    assert all(event["evidence"] for event in accepted)

    qa = client.get("/analysis/mission-graph/4d/qa")
    assert qa.status_code == 200
    rows = qa.json()["qa"]
    assert rows
    assert all(
        row["evidence_event_ids"] for row in rows if row["verification_status"] == "accepted"
    )

    evidence = client.get(
        "/analysis/mission-graph/4d/evidence",
        params={"event_id": accepted[0]["event_id"]},
    )
    assert evidence.status_code == 200
    assert evidence.json()["evidence"]
    missing_query = client.get("/analysis/mission-graph/4d/evidence")
    assert missing_query.status_code == 400
    missing_event = client.get(
        "/analysis/mission-graph/4d/evidence", params={"event_id": "evt-missing"}
    )
    assert missing_event.status_code == 404
