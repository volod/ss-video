"""Strict verifier, graph-program QA, and fail-closed review."""

from selfsuvis.pipeline.analysis4d.io import read_jsonl, read_model
from selfsuvis.pipeline.analysis4d.qa import execute_graph_program, generate_qa
from selfsuvis.pipeline.analysis4d.review import (
    ReviewDecision,
    ReviewError,
    ScriptedReview,
    parse_review_payload,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    SCHEMA_PROPOSAL,
    EvidenceRef,
    Proposal,
    QaAnswer,
    TimelineEvent,
    Verification,
)
from selfsuvis.pipeline.analysis4d.verifier import (
    publishable_events,
    verify_proposals,
)
from selfsuvis.pipeline.analysis4d.verifier_benchmark import _fixture_root, _load_geometry
from selfsuvis.pipeline.storage.analysis4d import publishable_metadata


def _bundle():
    root = _fixture_root()
    from selfsuvis.pipeline.analysis4d.schemas import GraphDelta, TrackRecord

    return {
        "tracks": read_jsonl(root / "tracks.jsonl", TrackRecord),
        "deltas": read_jsonl(root / "graph-deltas.jsonl", GraphDelta),
        "samples": _load_geometry(root),
        "proposals": read_jsonl(root / "proposals.jsonl", Proposal),
    }


def _relation(proposal_id: str, predicate: str, text: str, status: str = "accepted") -> Proposal:
    return Proposal(
        schema_version=SCHEMA_PROPOSAL,
        mission_id="mission-contradiction",
        proposal_id=proposal_id,
        claim_kind="relation",
        text=text,
        subject_id="n-a",
        predicate=predicate,
        object_id="n-b",
        start_sec=0.0,
        end_sec=1.0,
        verification_status=status,  # type: ignore[arg-type]
    )


def _verify(proposals, **kwargs):
    bundle = _bundle()
    return verify_proposals(
        proposals,
        tracks=bundle["tracks"],
        deltas=bundle["deltas"],
        samples=bundle["samples"],
        metric=True,
        frame="mission_enu",
        **kwargs,
    )


def test_geometry_rejects_a_confident_false_relation() -> None:
    output = _verify([_relation("bad", "right_of", "n-a is right of n-b confidence 0.99")])
    row = output.resolutions[0]
    assert row.status == "rejected"
    assert "geometry_contradiction" in row.reasons
    assert "geometry_overrides_confidence" in row.reasons


def test_geometry_accepts_left_of() -> None:
    output = _verify([_relation("good", "left_of", "n-a is left of n-b", status="uncertain")])
    assert output.resolutions[0].status == "accepted"
    assert "geometry" in output.resolutions[0].rules


def test_high_residual_stays_uncertain() -> None:
    bundle = _bundle()
    noisy = [sample.model_copy(update={"residual_m": 2.0}) for sample in bundle["samples"]]
    output = verify_proposals(
        [_relation("noisy", "left_of", "n-a is left of n-b")],
        tracks=bundle["tracks"],
        deltas=bundle["deltas"],
        samples=noisy,
        metric=True,
        frame="mission_enu",
    )
    assert output.resolutions[0].status == "uncertain"
    assert output.resolutions[0].reasons == ["uncertainty_gate"]


def test_disjoint_lifetime_is_rejected() -> None:
    proposal = _relation("late", "left_of", "n-a is left of n-b").model_copy(
        update={"start_sec": 10.0, "end_sec": 11.0}
    )
    output = _verify([proposal])
    assert output.resolutions[0].status == "rejected"
    assert "lifetime_disjoint" in output.resolutions[0].reasons


def test_track_count_accepts_two_crates_and_rejects_three() -> None:
    bundle = _bundle()
    output = _verify(
        [
            Proposal(
                schema_version=SCHEMA_PROPOSAL,
                mission_id="mission-contradiction",
                proposal_id="count-yes",
                claim_kind="attribute",
                text="two crates",
                verification_status="uncertain",
            ),
            Proposal(
                schema_version=SCHEMA_PROPOSAL,
                mission_id="mission-contradiction",
                proposal_id="count-no",
                claim_kind="attribute",
                text="three crates",
                verification_status="accepted",
            ),
        ]
    )
    by_id = {item.proposal_id: item for item in output.resolutions}
    assert by_id["count-yes"].status == "accepted"
    assert by_id["count-no"].status == "rejected"
    del bundle


def test_identity_without_a_link_stays_uncertain() -> None:
    output = _verify(
        [
            Proposal(
                schema_version=SCHEMA_PROPOSAL,
                mission_id="mission-contradiction",
                proposal_id="same",
                claim_kind="attribute",
                text="n-a is the same object as n-b",
                verification_status="accepted",
            )
        ]
    )
    assert output.resolutions[0].status == "uncertain"
    assert "vlm_not_sole_evidence" in output.resolutions[0].reasons


def test_action_needs_a_deterministic_interval() -> None:
    action = Proposal(
        schema_version=SCHEMA_PROPOSAL,
        mission_id="mission-contradiction",
        proposal_id="action-1",
        claim_kind="action",
        text="n-a was lifted",
        subject_id="n-a",
        start_sec=0.0,
        end_sec=1.0,
        verification_status="uncertain",
    )
    reviewer = ScriptedReview(
        decisions=[
            ReviewDecision(proposal_id="action-1", status="accepted", reasons=["visible lift"])
        ]
    )
    alone = _verify([action], reviewer=reviewer)
    assert alone.resolutions[0].status == "uncertain"
    covered = TimelineEvent(
        event_id="evt-count-crate-0000000",
        type="count_changed",
        summary="crate count changed to 2",
        start_sec=0.0,
        end_sec=1.0,
        participants=["track-a"],
        confidence=0.9,
        verification=Verification(status="accepted", rules=["track_count"]),
        evidence=[
            EvidenceRef(
                frame_id="mission-contradiction:track-a:0000000",
                t_sec=0.0,
                track_ids=["track-a"],
                geometry_ref="geometry/n-a/0.json",
            )
        ],
    )
    output = _verify([action], reviewer=reviewer, events=[covered])
    assert output.resolutions[0].status == "accepted"
    assert "deterministic_event" in output.resolutions[0].rules
    semantic = Proposal(
        schema_version=SCHEMA_PROPOSAL,
        mission_id="mission-contradiction",
        proposal_id="semantic",
        claim_kind="attribute",
        text="weathered crate",
        verification_status="accepted",
    )
    output = _verify(
        [_relation("good", "left_of", "n-a is left of n-b"), semantic],
        reviewer=ScriptedReview(error=ReviewError("timeout", "timed out")),
    )
    by_id = {item.proposal_id: item for item in output.resolutions}
    assert output.review_failure == "timeout"
    assert by_id["good"].status == "accepted"
    assert by_id["semantic"].status == "uncertain"
    assert "provider_unavailable" in output.degradations


def test_malformed_review_json_fails_closed() -> None:
    try:
        parse_review_payload('{"decisions": [{"proposal_id": "a", "extra": 1}]}')
    except ReviewError as exc:
        assert exc.code == "malformed"
    else:
        raise AssertionError("unknown field was accepted")


def test_remote_schema_parses_a_decision() -> None:
    payload = (
        '{"decisions":[{"proposal_id":"p-1","status":"uncertain","reasons":["disagree"],'
        '"corrected_text":null,"corrected_predicate":null}]}'
    )
    rows = parse_review_payload(payload)
    assert rows == [
        ReviewDecision(
            proposal_id="p-1",
            status="uncertain",
            reasons=["disagree"],
            corrected_text=None,
            corrected_predicate=None,
        )
    ]


def test_ambiguous_and_causal_programs_do_not_answer() -> None:
    event = TimelineEvent(
        event_id="evt-1",
        type="entered_region",
        summary="entered",
        start_sec=1.0,
        end_sec=2.0,
        participants=["track-a", "region-1"],
        confidence=0.9,
        verification=Verification(status="accepted", rules=["contains"]),
        evidence=[
            EvidenceRef(frame_id="mission:track-a:0001000", t_sec=1.0, track_ids=["track-a"])
        ],
    )
    other = event.model_copy(
        update={
            "event_id": "evt-2",
            "participants": ["track-b", "region-1"],
        }
    )
    assert (
        execute_graph_program(
            "entered(?track, region-1, after=0)", events=[event, other], tracks=[], edges=[]
        )
        is None
    )
    assert (
        execute_graph_program("caused(?track, region-1)", events=[event], tracks=[], edges=[])
        is None
    )
    unique = execute_graph_program(
        "entered(?track, region-1, after=0)", events=[event], tracks=[], edges=[]
    )
    assert unique is not None
    assert unique.answer == QaAnswer(kind="track_ref", value="track-a", unit=None)
    assert generate_qa("mission-contradiction", [event], [], [])[0].evidence_event_ids == ["evt-1"]


def test_publishable_rows_omit_rejected_claims() -> None:
    accepted = TimelineEvent(
        event_id="evt-ok",
        type="count_changed",
        summary="crate count changed to 2",
        start_sec=0.0,
        end_sec=1.0,
        participants=["track-a"],
        confidence=0.9,
        verification=Verification(status="accepted", rules=["track_count"]),
        evidence=[
            EvidenceRef(
                frame_id="mission:track-a:0000000",
                t_sec=0.0,
                track_ids=["track-a"],
                geometry_ref="geometry/n-a/0.json",
            )
        ],
    )
    rejected = accepted.model_copy(
        update={
            "event_id": "evt-no",
            "verification": Verification(status="rejected", rules=["geometry"], reasons=["no"]),
        }
    )
    assert [event.event_id for event in publishable_events([accepted, rejected])] == ["evt-ok"]
    rows = publishable_metadata(
        {
            "run": {"id": "run"},
            "events": [
                {"verification_status": "accepted"},
                {"verification_status": "uncertain"},
            ],
            "edges": [
                {"verification_status": "accepted"},
                {"verification_status": "rejected"},
            ],
            "qa": [
                {"verification_status": "accepted"},
                {"verification_status": "rejected"},
            ],
        }
    )
    assert rows["events"] == [{"verification_status": "accepted"}]
    assert rows["edges"] == [{"verification_status": "accepted"}]
    assert rows["qa"] == [{"verification_status": "accepted"}]


def test_workflow_writes_evidence_linked_qa(tmp_path) -> None:
    import shutil

    from selfsuvis.pipeline.analysis4d.validate import validate_bundle
    from selfsuvis.pipeline.workflows.analysis4d_verify import run_mission_verify

    dest = tmp_path / "4d"
    shutil.copytree(_fixture_root(), dest)
    from selfsuvis.pipeline.analysis4d.store import AnalysisStore

    AnalysisStore(dest).append_proposal(
        Proposal(
            schema_version=SCHEMA_PROPOSAL,
            mission_id="mission-contradiction",
            proposal_id="proposal-semantic",
            claim_kind="attribute",
            text="weathered crate",
            verification_status="accepted",
        )
    )
    result = run_mission_verify(
        "mission-contradiction",
        dest=dest,
        reviewer=ScriptedReview(error=ReviewError("refusal", "refused")),
    )
    bundle = validate_bundle(dest)
    assert result.review_failure == "refusal"
    assert all(event.evidence for event in result.events if event.verification.status == "accepted")
    assert all(row.evidence_event_ids for row in result.qa if row.verification_status == "accepted")
    assert bundle.timeline.qa_pairs
    again = run_mission_verify("mission-contradiction", dest=dest)
    assert [event.event_id for event in again.events] == [event.event_id for event in result.events]
    timeline = read_model(dest / "timeline.json", type(bundle.timeline))
    assert timeline.supersedes_sha256
