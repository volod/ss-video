"""Temporal scene graph reducer, proposals, and the pinned benchmark."""

from selfsuvis.pipeline.analysis4d.graph import RegionBox, materialize, reduce_graph
from selfsuvis.pipeline.analysis4d.graph_benchmark import run_graph_benchmark
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_GEOMETRY,
    SCHEMA_TRACK,
    Box2D,
    GeometrySample,
    IdentityLink,
    TrackRecord,
)
from selfsuvis.pipeline.analysis4d.vlm import (
    ScriptedVlm,
    VlmClaim,
    parse_vlm_payload,
    proposals_from_claims,
)
from selfsuvis.pipeline.workflows.analysis4d_graph import run_mission_graph


def test_graph_benchmark_gates(tmp_path) -> None:
    report = run_graph_benchmark(tmp_path, probe_gpu=False)
    assert report.failures == []
    assert report.passed
    assert report.replay_deterministic
    assert report.intervals_valid
    assert report.vlm_degraded
    assert report.deterministic_edge_count > 0
    assert report.relation_precision is not None
    assert report.yolo_relation_precision is not None
    assert report.relation_precision > report.yolo_relation_precision


def test_replay_ignores_input_order() -> None:
    tracks = [
        _track("obs-1", "track-1", 0.0),
        _track("obs-2", "track-3", 2.0, link="track-1"),
    ]
    reduced = reduce_graph("mission-id", tracks, [], frame_name="mission_enu", metric=False)
    forward = materialize(reduced.deltas)
    backward = materialize(list(reversed(reduced.deltas)))
    assert [node.node_id for node in forward.nodes] == ["track-3"]
    assert [node.node_id for node in forward.superseded_nodes] == ["track-1"]
    assert [node.node_id for node in backward.nodes] == [node.node_id for node in forward.nodes]
    assert forward.superseded_delta_ids == backward.superseded_delta_ids


def test_unavailable_vlm_keeps_deterministic_edges(tmp_path) -> None:
    from selfsuvis.pipeline.analysis4d.graph_benchmark import prepare_pinned_scene
    from selfsuvis.pipeline.analysis4d.vlm import UnavailableVlm

    prepare_pinned_scene(tmp_path)
    result = run_mission_graph(
        "mission-graph",
        dest=tmp_path,
        provider=UnavailableVlm(),
        regions=[RegionBox("region-1", "loading-zone", [0.0, 0.0, 0.0], [10.0, 10.0, 4.0])],
    )
    assert "provider_unavailable" in result.degradations
    assert result.view.edges
    assert result.proposals == []
    assert any(event.type == "entered_region" for event in result.events)
    again = run_mission_graph("mission-graph", dest=tmp_path, provider=UnavailableVlm())
    assert len(again.deltas) == len(result.deltas)


def test_scripted_vlm_keeps_rejected_corrected_and_superseded_claims(tmp_path) -> None:
    from selfsuvis.pipeline.analysis4d.graph_benchmark import prepare_pinned_scene

    prepare_pinned_scene(tmp_path)
    claims = [
        VlmClaim(
            claim_kind="attribute",
            text="crate is a crate",
            proposal_id="proposal-open",
            verification_status="uncertain",
        ),
        VlmClaim(
            claim_kind="relation",
            text="track-a is right of track-b",
            proposal_id="proposal-false",
            subject_id="track-a",
            predicate="right_of",
            object_id="track-b",
            start_sec=0.0,
            end_sec=1.0,
            verification_status="uncertain",
        ),
        VlmClaim(
            claim_kind="attribute",
            text="crate color corrected",
            proposal_id="proposal-fix",
            verification_status="corrected",
            supersedes="proposal-open",
        ),
    ]
    result = run_mission_graph(
        "mission-graph",
        dest=tmp_path,
        provider=ScriptedVlm(claims),
        regions=[RegionBox("region-1", "loading-zone", [0.0, 0.0, 0.0], [10.0, 10.0, 4.0])],
    )
    by_id = {row.proposal_id: row for row in result.proposals}
    assert by_id["proposal-open"].verification_status == "uncertain"
    assert by_id["proposal-false"].verification_status == "rejected"
    assert by_id["proposal-fix"].verification_status == "corrected"
    assert by_id["proposal-fix"].supersedes == "proposal-open"
    assert result.view.edges


def test_malformed_vlm_json_fails_closed() -> None:
    try:
        parse_vlm_payload('[{"claim_kind": "attribute", "text": "x", "extra": 1}]')
    except ValueError as exc:
        assert "unknown_field" in str(exc)
    else:
        raise AssertionError("extra field was accepted")


def test_support_interval_emits_put_down_and_picked_up() -> None:
    tracks = [_track("obs-1", "track-box", 0.0), _track("obs-2", "track-box", 2.0)]
    samples = [
        _sample("ground", 0.0, [0.0, 0.0, -0.2], [4.0, 4.0, 0.4]),
        _sample("ground", 2.0, [0.0, 0.0, -0.2], [4.0, 4.0, 0.4]),
        _sample("track-box", 0.0, [0.0, 0.0, 0.25], [0.4, 0.4, 0.4]),
        _sample("track-box", 2.0, [0.0, 0.0, 1.2], [0.4, 0.4, 0.4]),
    ]
    reduced = reduce_graph(
        "mission-id",
        tracks,
        samples,
        frame_name="mission_enu",
        metric=True,
        region_ids={"ground"},
    )
    types = {event.type for event in reduced.events}
    assert "put_down" in types
    assert "picked_up" in types


def test_proposals_reject_geometry_contradictions() -> None:
    sample_a = _sample("track-a", 0.0, [1.0, 0.0, 0.5], [0.4, 0.4, 0.4])
    sample_b = _sample("track-b", 0.0, [3.0, 0.0, 0.5], [0.4, 0.4, 0.4])
    claims = [
        VlmClaim(
            claim_kind="relation",
            text="a is right of b",
            proposal_id="proposal-false",
            subject_id="track-a",
            predicate="right_of",
            object_id="track-b",
            start_sec=0.0,
            end_sec=1.0,
        )
    ]
    rows = proposals_from_claims("mission-id", claims, [sample_a, sample_b])
    assert rows[0].verification_status == "rejected"
    assert "geometry_contradiction" in rows[0].reasons


def _track(
    observation_id: str, track_id: str, t_sec: float, link: str | None = None
) -> TrackRecord:
    return TrackRecord(
        schema_version=SCHEMA_TRACK,
        mission_id="mission-id",
        observation_id=observation_id,
        track_id=track_id,
        state="confirmed",
        t_sec=t_sec,
        prompt_id="prompt-crate",
        label_raw="crate",
        label_normalized="crate",
        box=Box2D(xywh_norm=[0.1, 0.1, 0.2, 0.2]),
        confidence=0.9,
        identity_link=None if link is None else IdentityLink(linked_track_id=link, reason="cut"),
    )


def _sample(subject: str, t_sec: float, center: list[float], extent: list[float]) -> GeometrySample:
    millis = int(round(t_sec * 1000))
    return GeometrySample(
        schema_version=SCHEMA_GEOMETRY,
        mission_id="mission-id",
        sample_id=f"geo-{subject}-{millis:07d}",
        subject_id=subject,
        t_sec=t_sec,
        frame="mission_enu",
        center_m=center,
        extent_m=extent,
        quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        metric_scale="metric",
        calibration_id="cal-bench",
    )
