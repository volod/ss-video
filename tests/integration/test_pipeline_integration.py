"""Integration tests for the production pipeline.

Exercises multi-module interactions without live Docker, GPU, or file I/O beyond
tmp_path fixtures:

- Active-learning tagging → frame persistence → DB verification
- Change detection across GPS-overlapping missions (single and multi-mission)
- Semantic diff populated when frame_facts_json is present
- Report generator: HTML written to disk, content validated
- Full worker job loop: create → claim → upsert_mission → replace_frames → finish
- AL-to-report round-trip: scores assigned → frames persisted → HTML generated
"""

import asyncio
import os
from typing import Any

import numpy as np
import pytest

from tests.support.db import PipelineMockConn, make_frame_record


def run(coro):
    return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_embedding(seed: int = 0, dim: int = 512) -> np.ndarray:
    """Return a deterministic, L2-normalised unit embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _frames_with_al(
    mission_id: str,
    n: int,
    top_k: int = 2,
    novel_threshold: float = 0.7,
) -> tuple[list[dict[str, Any]], list[float], list[str]]:
    """Create *n* synthetic frames, assign AL tags, and return (frame_records, scores, tags)."""
    from selfsuvis.pipeline.analysis.active_learning import assign_al_tags

    dino_dists = [float(i) / (n - 1) if n > 1 else 0.5 for i in range(n)]
    cap_confs = [1.0 - d for d in dino_dists]
    scores, tags = assign_al_tags(
        dino_dists, cap_confs, top_k=top_k, novel_threshold=novel_threshold
    )

    frames = []
    for i, (score, tag) in enumerate(zip(scores, tags)):
        rec = make_frame_record(
            frame_id=f"{mission_id}_f{i}",
            mission_id=mission_id,
            t_sec=float(i * 5),
            gps_lat=47.0,
            gps_lon=8.0,
        )
        rec["al_score"] = score
        rec["al_tag"] = tag
        frames.append(rec)

    return frames, scores, tags


# ══════════════════════════════════════════════════════════════════════════════
# 1. Active learning → frame persistence
# ══════════════════════════════════════════════════════════════════════════════


def test_al_tags_persisted_correctly():
    """AL tags computed by assign_al_tags survive the round-trip through replace_frames."""
    from selfsuvis.pipeline.storage.missions import replace_frames

    conn = PipelineMockConn()
    frames, scores, tags = _frames_with_al("m1", n=6, top_k=2)
    run(replace_frames(conn, "m1", frames))

    assert len(conn._frames) == 6

    # Top-2 by score must be tagged needs_annotation
    sorted_frames = sorted(conn._frames.values(), key=lambda f: f["al_score"], reverse=True)
    top2 = sorted_frames[:2]
    for f in top2:
        assert f["al_tag"] == "needs_annotation", (
            f"Frame {f['id']} has score={f['al_score']:.3f} but tag={f['al_tag']}"
        )


def test_al_tags_none_for_low_uncertainty():
    """Frames with low dino_dist and high caption confidence get tag='none'."""
    from selfsuvis.pipeline.analysis.active_learning import assign_al_tags
    from selfsuvis.pipeline.storage.missions import replace_frames

    conn = PipelineMockConn()
    # All frames highly certain: dino_dist≈0, confidence≈1
    n = 5
    dino_dists = [0.01] * n
    cap_confs = [0.99] * n
    scores, tags = assign_al_tags(dino_dists, cap_confs, top_k=2, novel_threshold=0.7)

    frames = []
    for i, (score, tag) in enumerate(zip(scores, tags)):
        rec = make_frame_record(f"m2_f{i}", "m2", t_sec=float(i))
        rec["al_score"] = score
        rec["al_tag"] = tag
        frames.append(rec)

    run(replace_frames(conn, "m2", frames))

    # No frames should be novel (all dino_dist < 0.7)
    novel_frames = [f for f in conn._frames.values() if f["al_tag"] == "novel"]
    assert novel_frames == []


def test_al_novel_tag_assigned_above_threshold():
    """Frames with dino_dist ≥ novel_threshold (outside top-K) get tag='novel'."""
    from selfsuvis.pipeline.analysis.active_learning import assign_al_tags
    from selfsuvis.pipeline.storage.missions import replace_frames

    conn = PipelineMockConn()
    # Frame 0: top-K candidate (high score), frame 1: high dist but rank 2+ → novel
    dino_dists = [0.9, 0.85, 0.0, 0.0, 0.0]
    cap_confs = [0.1, 0.5, 0.95, 0.95, 0.95]

    scores, tags = assign_al_tags(dino_dists, cap_confs, top_k=1, novel_threshold=0.7)

    frames = [
        {
            **make_frame_record(f"m3_f{i}", "m3", t_sec=float(i)),
            "al_score": scores[i],
            "al_tag": tags[i],
        }
        for i in range(5)
    ]
    run(replace_frames(conn, "m3", frames))

    tags_stored = {f["id"]: f["al_tag"] for f in conn._frames.values()}
    # At least one frame should be novel
    assert "novel" in tags_stored.values()


def test_al_score_ordering_matches_score_field():
    """al_score values in DB match what assign_al_tags computed."""
    from selfsuvis.pipeline.analysis.active_learning import assign_al_tags
    from selfsuvis.pipeline.storage.missions import replace_frames

    conn = PipelineMockConn()
    dino_dists = [0.1, 0.5, 0.9]
    cap_confs = [0.9, 0.5, 0.1]
    scores, tags = assign_al_tags(dino_dists, cap_confs, top_k=1)

    frames = [
        {
            **make_frame_record(f"m4_f{i}", "m4", t_sec=float(i)),
            "al_score": scores[i],
            "al_tag": tags[i],
        }
        for i in range(3)
    ]
    run(replace_frames(conn, "m4", frames))

    for i in range(3):
        stored = conn._frames[f"m4_f{i}"]
        assert stored["al_score"] == pytest.approx(scores[i], abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Change detection
# ══════════════════════════════════════════════════════════════════════════════


def _make_cd_frame(
    frame_id: str,
    mission_id: str,
    lat: float = 47.0,
    lon: float = 8.0,
    seed: int = 0,
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Minimal frame dict for change detection."""
    return {
        "frame_id": frame_id,
        "mission_id": mission_id,
        "embedding": _make_embedding(seed).tolist(),
        "gps": {"lat": lat, "lon": lon},
        "frame_facts_json": facts,
    }


def _ref_query_fn(ref_frames: list[dict[str, Any]]):
    """Synthetic query_fn that returns all ref_frames regardless of bbox."""

    def _fn(embedding, bbox):
        return ref_frames

    return _fn


def test_detect_changes_across_missions():
    """Two missions at the same location with different embeddings → change detected."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    ref = _make_cd_frame("ref_f1", "mission_old", seed=0)
    new = _make_cd_frame("new_f1", "mission_new", seed=99)  # orthogonal → high dist

    changes = detect_changes(
        new_frames=[new],
        query_fn=_ref_query_fn([ref]),
        threshold=0.01,  # low threshold: any difference triggers
    )

    assert len(changes) == 1
    assert changes[0]["frame_id"] == "new_f1"
    assert changes[0]["mission_id"] == "mission_new"
    assert changes[0]["ref_frame_id"] == "ref_f1"
    assert changes[0]["ref_mission_id"] == "mission_old"
    assert changes[0]["change_score"] > 0.0


def test_detect_changes_skips_same_mission_frames():
    """Frames the same mission are never flagged as changes."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    ref = _make_cd_frame("f_old", "same_mission", seed=0)
    new = _make_cd_frame("f_new", "same_mission", seed=99)

    changes = detect_changes(
        new_frames=[new],
        query_fn=_ref_query_fn([ref]),
        threshold=0.01,
    )

    assert changes == []


def test_detect_changes_no_gps_frame_skipped():
    """Frames without GPS coordinates are silently skipped."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    new_frame = {
        "frame_id": "no_gps",
        "mission_id": "m_new",
        "embedding": _make_embedding(1).tolist(),
        "gps": None,
    }
    ref = _make_cd_frame("ref_f", "m_old", seed=0)

    changes = detect_changes(
        new_frames=[new_frame],
        query_fn=_ref_query_fn([ref]),
        threshold=0.01,
    )

    assert changes == []


def test_detect_changes_below_threshold_no_event():
    """Identical embeddings → cosine distance ≈ 0 → no change event."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    emb = _make_embedding(42).tolist()
    ref = {
        "frame_id": "ref",
        "mission_id": "m_old",
        "embedding": emb,
        "gps": {"lat": 47.0, "lon": 8.0},
    }
    new = {
        "frame_id": "new",
        "mission_id": "m_new",
        "embedding": emb,
        "gps": {"lat": 47.0, "lon": 8.0},
    }

    changes = detect_changes(
        new_frames=[new],
        query_fn=_ref_query_fn([ref]),
        threshold=0.5,  # high threshold; identical embeddings won't reach it
    )

    assert changes == []


def test_detect_changes_multiple_refs_picks_closest():
    """With multiple reference frames detect_changes picks the closest one."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    emb_new = _make_embedding(42)
    # ref_a: orthogonal to emb_new (high distance)
    emb_a = _make_embedding(99)
    # ref_b: close to emb_new (small distance, below threshold)
    # Construct ref_b ≈ emb_new but slightly perturbed
    emb_b = emb_new + 0.001 * np.ones_like(emb_new)
    emb_b = (emb_b / np.linalg.norm(emb_b)).tolist()

    refs = [
        {
            "frame_id": "ref_a",
            "mission_id": "m_old",
            "embedding": emb_a.tolist(),
            "gps": {"lat": 47.0, "lon": 8.0},
        },
        {
            "frame_id": "ref_b",
            "mission_id": "m_old",
            "embedding": emb_b,
            "gps": {"lat": 47.0, "lon": 8.0},
        },
    ]
    new = {
        "frame_id": "new",
        "mission_id": "m_new",
        "embedding": emb_new.tolist(),
        "gps": {"lat": 47.0, "lon": 8.0},
    }

    changes = detect_changes(
        new_frames=[new],
        query_fn=_ref_query_fn(refs),
        threshold=0.5,  # below ref_a's dist, above ref_b's dist
    )

    # ref_b is closest (low dist) → below threshold → no event
    assert changes == []


def test_detect_changes_empty_new_frames():
    """No new frames → no changes."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    changes = detect_changes(
        new_frames=[],
        query_fn=_ref_query_fn([_make_cd_frame("ref", "m_old")]),
        threshold=0.01,
    )

    assert changes == []


def test_detect_changes_no_references_no_event():
    """No reference frames in the query_fn → nothing to compare → no change."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    new = _make_cd_frame("new_f", "m_new", seed=1)

    changes = detect_changes(
        new_frames=[new],
        query_fn=_ref_query_fn([]),  # empty candidates
        threshold=0.01,
    )

    assert changes == []


def test_detect_changes_change_score_in_event():
    """Change event carries the cosine distance as change_score."""
    from selfsuvis.pipeline.analysis.change_detection import cosine_distance, detect_changes

    emb_new = _make_embedding(0)
    emb_ref = _make_embedding(99)
    expected_dist = cosine_distance(emb_new, emb_ref)

    ref = {
        "frame_id": "ref",
        "mission_id": "m_old",
        "embedding": emb_ref.tolist(),
        "gps": {"lat": 47.0, "lon": 8.0},
    }
    new = {
        "frame_id": "new",
        "mission_id": "m_new",
        "embedding": emb_new.tolist(),
        "gps": {"lat": 47.0, "lon": 8.0},
    }

    changes = detect_changes(
        new_frames=[new],
        query_fn=_ref_query_fn([ref]),
        threshold=0.01,
    )

    assert len(changes) == 1
    assert changes[0]["change_score"] == pytest.approx(expected_dist, rel=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Semantic diff
# ══════════════════════════════════════════════════════════════════════════════


def test_semantic_diff_vehicle_count_change():
    """compute_semantic_diff detects a change in vehicle count."""
    from selfsuvis.pipeline.analysis.change_detection import compute_semantic_diff

    ref_facts = {"vehicle_groups": [{"count": 2, "type": "car"}]}
    new_facts = {"vehicle_groups": [{"count": 5, "type": "car"}]}

    diff = compute_semantic_diff(ref_facts, new_facts)

    assert "vehicle_count" in diff
    assert diff["vehicle_count"]["before"] == 2
    assert diff["vehicle_count"]["after"] == 5
    assert diff["vehicle_count"]["delta"] == 3


def test_semantic_diff_road_condition_change():
    """compute_semantic_diff detects road_condition transitions."""
    from selfsuvis.pipeline.analysis.change_detection import compute_semantic_diff

    diff = compute_semantic_diff(
        {"road_condition": "clear"},
        {"road_condition": "wet"},
    )

    assert "road_condition" in diff
    assert diff["road_condition"]["before"] == "clear"
    assert diff["road_condition"]["after"] == "wet"


def test_semantic_diff_no_change_empty():
    """Identical facts → empty diff dict."""
    from selfsuvis.pipeline.analysis.change_detection import compute_semantic_diff

    facts = {
        "vehicle_groups": [{"count": 3}],
        "road_condition": "clear",
        "road_surface": "asphalt",
    }
    diff = compute_semantic_diff(facts, facts)

    assert diff == {}


def test_detect_changes_populates_semantic_diff():
    """When both frames have frame_facts_json, semantic_diff_json is set on the event."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    ref_facts = {"vehicle_groups": [{"count": 1}], "road_condition": "clear"}
    new_facts = {"vehicle_groups": [{"count": 4}], "road_condition": "wet"}

    ref = _make_cd_frame("ref", "m_old", seed=0, facts=ref_facts)
    new = _make_cd_frame("new", "m_new", seed=99, facts=new_facts)

    changes = detect_changes(
        new_frames=[new],
        query_fn=_ref_query_fn([ref]),
        threshold=0.01,
    )

    assert len(changes) == 1
    sdiff = changes[0]["semantic_diff_json"]
    assert sdiff is not None
    assert "vehicle_count" in sdiff
    assert "road_condition" in sdiff


def test_detect_changes_no_facts_semantic_diff_none():
    """When facts are missing, semantic_diff_json is None."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes

    ref = _make_cd_frame("ref", "m_old", seed=0)  # no facts
    new = _make_cd_frame("new", "m_new", seed=99)  # no facts

    changes = detect_changes(
        new_frames=[new],
        query_fn=_ref_query_fn([ref]),
        threshold=0.01,
    )

    assert len(changes) == 1
    assert changes[0]["semantic_diff_json"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 4. Report generator
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def report_dir(tmp_path, monkeypatch):
    """Point DATA_DIR at tmp_path so report files land somewhere writable."""
    import selfsuvis.pipeline.core.config as cfg

    monkeypatch.setattr(cfg.settings, "DATA_DIR", str(tmp_path))
    return tmp_path


def _sample_frames(mission_id: str = "m1") -> list[dict[str, Any]]:
    return [
        {
            "frame_path": "/frames/f0.jpg",
            "caption": "road ahead",
            "al_tag": "none",
            "al_score": 0.1,
            "t_sec": 0.0,
        },
        {
            "frame_path": "/frames/f1.jpg",
            "caption": "construction zone",
            "al_tag": "needs_annotation",
            "al_score": 0.85,
            "t_sec": 5.0,
        },
        {
            "frame_path": "/frames/f2.jpg",
            "caption": "unusual vehicle",
            "al_tag": "novel",
            "al_score": 0.72,
            "t_sec": 10.0,
        },
    ]


def test_report_writer_creates_file(report_dir):
    """write_mission_report creates summary.html under DATA_DIR/reports/mission_id/."""
    from selfsuvis.pipeline.workflows.reporting import write_mission_report

    path = write_mission_report("mission-42", _sample_frames())

    assert os.path.isfile(path)
    assert path.endswith("summary.html")
    assert "mission-42" in path


def test_report_html_contains_mission_id(report_dir):
    """Generated HTML references the mission_id in the <title> and <h1>."""
    from selfsuvis.pipeline.workflows.reporting import write_mission_report

    write_mission_report("mission-X", _sample_frames())
    html_path = os.path.join(str(report_dir), "reports", "mission-X", "summary.html")
    content = open(html_path).read()

    assert "mission-X" in content


def test_report_al_tag_distribution(report_dir):
    """HTML contains correct counts for each AL tag."""
    from selfsuvis.pipeline.workflows.reporting import write_mission_report

    write_mission_report("m-dist", _sample_frames())
    html_path = os.path.join(str(report_dir), "reports", "m-dist", "summary.html")
    content = open(html_path).read()

    # Needs annotation: 1, novel: 1, none: 1
    assert "Needs annotation: 1" in content
    assert "Novel: 1" in content
    assert "None: 1" in content


def test_report_frame_count_in_html(report_dir):
    """HTML reports the correct total frame count."""
    from selfsuvis.pipeline.workflows.reporting import write_mission_report

    frames = _sample_frames()
    write_mission_report("m-cnt", frames)
    html_path = os.path.join(str(report_dir), "reports", "m-cnt", "summary.html")
    content = open(html_path).read()

    assert f"Frames: {len(frames)}" in content


def test_report_duration_in_html(report_dir):
    """HTML shows the max t_sec as duration."""
    from selfsuvis.pipeline.workflows.reporting import write_mission_report

    write_mission_report("m-dur", _sample_frames())
    html_path = os.path.join(str(report_dir), "reports", "m-dur", "summary.html")
    content = open(html_path).read()

    # max t_sec == 10.0
    assert "10.0s" in content


def test_report_captions_escaped(report_dir):
    """Captions with HTML-special characters are escaped."""
    from selfsuvis.pipeline.workflows.reporting import write_mission_report

    frames = [
        {
            "frame_path": "/f.jpg",
            "caption": "<script>alert(1)</script>",
            "al_tag": "none",
            "al_score": 0.0,
            "t_sec": 0.0,
        }
    ]
    write_mission_report("m-xss", frames)
    html_path = os.path.join(str(report_dir), "reports", "m-xss", "summary.html")
    content = open(html_path).read()

    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_report_sorted_by_score_descending(report_dir):
    """Frames appear in the HTML ordered by al_score descending."""
    from selfsuvis.pipeline.workflows.reporting import generate_summary_html

    frames = [
        {
            "frame_path": "/frames/low.jpg",
            "caption": "low",
            "al_tag": "none",
            "al_score": 0.1,
            "t_sec": 0.0,
        },
        {
            "frame_path": "/frames/high.jpg",
            "caption": "high",
            "al_tag": "needs_annotation",
            "al_score": 0.9,
            "t_sec": 1.0,
        },
    ]
    html = generate_summary_html("m-sort", frames)

    # high.jpg must appear before low.jpg
    assert html.index("high.jpg") < html.index("low.jpg")


def test_report_empty_frames(report_dir):
    """write_mission_report handles an empty frame list gracefully."""
    from selfsuvis.pipeline.workflows.reporting import write_mission_report

    path = write_mission_report("m-empty", [])
    content = open(path).read()

    assert "Frames: 0" in content
    assert "Duration: 0.0s" in content


# ══════════════════════════════════════════════════════════════════════════════
# 5. Worker job loop
# ══════════════════════════════════════════════════════════════════════════════


def test_worker_job_loop_end_to_end():
    """Simulates the worker: create → claim → mission upsert → frames → finish."""
    from selfsuvis.pipeline.storage.jobs import create_job, fetch_and_claim_next_pending, update_job
    from selfsuvis.pipeline.storage.missions import (
        mark_mission_finished,
        replace_frames,
        upsert_mission,
    )

    conn = PipelineMockConn()

    # 1. API enqueues job
    run(create_job(conn, "job-1", {"video_id": "vid-1", "mission_id": "mis-1"}))
    assert conn._jobs["job-1"]["status"] == "pending"

    # 2. Worker claims it
    job = run(fetch_and_claim_next_pending(conn))
    assert job is not None
    assert job["id"] == "job-1"
    assert conn._jobs["job-1"]["status"] == "running"

    # 3. Worker creates mission record
    run(
        upsert_mission(
            conn,
            mission_id="mis-1",
            video_id="vid-1",
            video_path="/data/vid-1.mp4",
            job_id="job-1",
            robot_id="drone-a",
            status="indexing",
            frame_count=3,
            duration_sec=15.0,
            gps_origin={"lat": 47.0, "lon": 8.0, "alt": 400.0},
        )
    )
    assert conn._missions["mis-1"]["status"] == "indexing"

    # 4. Worker persists frames with AL tags
    frames, _, _ = _frames_with_al("mis-1", n=3, top_k=1)
    run(replace_frames(conn, "mis-1", frames))
    assert len(conn._frames) == 3

    # 5. Worker marks mission done
    run(mark_mission_finished(conn, "mis-1", status="done", pose_status="skipped"))
    assert conn._missions["mis-1"]["status"] == "done"

    # 6. Worker finalises job
    run(update_job(conn, "job-1", status="finished", progress={"frames": 3}))
    assert conn._jobs["job-1"]["status"] == "finished"
    assert conn._jobs["job-1"]["progress"]["frames"] == 3


def test_worker_marks_job_error_on_failure():
    """Worker records error status when indexing fails."""
    from selfsuvis.pipeline.storage.jobs import create_job, fetch_and_claim_next_pending, update_job
    from selfsuvis.pipeline.storage.missions import mark_mission_finished, upsert_mission

    conn = PipelineMockConn()

    run(create_job(conn, "job-err", {"video_id": "vid-err", "mission_id": "mis-err"}))
    run(fetch_and_claim_next_pending(conn))

    run(
        upsert_mission(
            conn,
            mission_id="mis-err",
            video_id="vid-err",
            video_path="/data/broken.mp4",
            job_id="job-err",
            robot_id="rover-b",
            status="indexing",
            frame_count=0,
            duration_sec=None,
            gps_origin=None,
        )
    )

    # Simulate ffmpeg crash
    run(mark_mission_finished(conn, "mis-err", status="error", error="ffmpeg exit code 1"))
    run(update_job(conn, "job-err", status="failed", error="ffmpeg exit code 1"))

    assert conn._missions["mis-err"]["status"] == "error"
    assert "ffmpeg" in (conn._missions["mis-err"].get("error") or "")
    assert conn._jobs["job-err"]["status"] == "failed"


def test_worker_second_job_waits_for_first():
    """Two pending jobs: first is claimed, second remains pending."""
    from selfsuvis.pipeline.storage.jobs import create_job, fetch_and_claim_next_pending

    conn = PipelineMockConn()
    run(create_job(conn, "job-a", {"video_id": "va"}))
    run(create_job(conn, "job-b", {"video_id": "vb"}))

    first = run(fetch_and_claim_next_pending(conn))
    assert first is not None
    assert first["id"] == "job-a"  # oldest job claimed first

    # job-b still pending
    assert conn._jobs["job-b"]["status"] == "pending"


def test_worker_no_pending_returns_none():
    """fetch_and_claim_next_pending returns None when queue is empty."""
    from selfsuvis.pipeline.storage.jobs import fetch_and_claim_next_pending

    conn = PipelineMockConn()
    result = run(fetch_and_claim_next_pending(conn))

    assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 6. AL → DB → report round-trip
# ══════════════════════════════════════════════════════════════════════════════


def test_al_to_report_round_trip(tmp_path, monkeypatch):
    """Full pipeline: AL scoring → DB persistence → HTML report generated correctly."""
    import selfsuvis.pipeline.core.config as cfg

    monkeypatch.setattr(cfg.settings, "DATA_DIR", str(tmp_path))

    from selfsuvis.pipeline.analysis.active_learning import assign_al_tags
    from selfsuvis.pipeline.storage.missions import replace_frames
    from selfsuvis.pipeline.workflows.reporting import generate_summary_html

    conn = PipelineMockConn()

    dino_dists = [0.1, 0.6, 0.95, 0.3, 0.8]
    cap_confs = [0.9, 0.4, 0.05, 0.7, 0.2]
    scores, tags = assign_al_tags(dino_dists, cap_confs, top_k=2, novel_threshold=0.7)

    frames = []
    for i, (score, tag) in enumerate(zip(scores, tags)):
        rec = make_frame_record(f"mis-rt_f{i}", "mis-rt", t_sec=float(i * 3))
        rec["al_score"] = score
        rec["al_tag"] = tag
        rec["caption"] = f"frame {i} caption"
        frames.append(rec)

    run(replace_frames(conn, "mis-rt", frames))

    # Verify DB state before report
    assert len(conn._frames) == 5
    annotate_count = sum(1 for f in conn._frames.values() if f["al_tag"] == "needs_annotation")
    assert annotate_count == 2

    # Build report DB frames
    db_frames = [
        {
            "frame_path": f["frame_path"],
            "caption": f.get("caption") or "",
            "al_tag": f["al_tag"],
            "al_score": f["al_score"],
            "t_sec": f["t_sec"],
        }
        for f in conn._frames.values()
    ]
    html = generate_summary_html("mis-rt", db_frames)

    assert "Needs annotation: 2" in html
    assert "mis-rt" in html
    # Highest-scoring frame appears first
    sorted_db = sorted(db_frames, key=lambda f: f["al_score"], reverse=True)
    top_path = sorted_db[0]["frame_path"]
    assert top_path in html


def test_multi_mission_change_detection_pipeline():
    """Two missions; detect_changes finds a change between them, respects mission boundary."""
    from selfsuvis.pipeline.analysis.change_detection import detect_changes
    from selfsuvis.pipeline.storage.missions import replace_frames

    conn = PipelineMockConn()

    # Mission 1: reference frames
    m1_frames, _, _ = _frames_with_al("m1", n=3)
    run(replace_frames(conn, "m1", m1_frames))

    # Mission 2: new frames for change detection (note: detect_changes needs
    # 'frame_id' key, not 'id', so we build plain dicts here)
    m2_frames = [
        {
            "frame_id": f"m2_f{i}",
            "mission_id": "m2",
            "embedding": _make_embedding(seed=i + 50).tolist(),
            "gps": {"lat": 47.0, "lon": 8.0},
        }
        for i in range(3)
    ]

    # Build reference pool m1 frames
    ref_pool = [
        {
            "frame_id": f["id"],
            "mission_id": "m1",
            "embedding": _make_embedding(seed=i).tolist(),
            "gps": {"lat": 47.0, "lon": 8.0},
        }
        for i, f in enumerate(m1_frames)
    ]

    changes = detect_changes(
        new_frames=m2_frames,
        query_fn=_ref_query_fn(ref_pool),
        threshold=0.01,  # catch any difference
    )

    # All 3 m2 frames are at the same GPS location as m1 and have different embeddings
    assert len(changes) == 3
    for c in changes:
        assert c["mission_id"] == "m2"
        assert c["ref_mission_id"] == "m1"
        assert c["change_score"] > 0.0
