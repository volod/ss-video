"""Integration tests for pipeline/mission_db.py.

Tests the full mission + frame persistence layer using the in-process
PipelineMockConn.  No live PostgreSQL required.

Covers:
- upsert_mission: create and idempotent update
- replace_frames: atomic clear-and-insert
- mark_mission_finished: status transitions
- apply_gps_registration: ENU origin + per-frame pose propagation
- list_frames_after: cursor-based pagination
"""

import asyncio

import pytest

from tests.support.db import PipelineMockConn, make_frame_record


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def conn():
    return PipelineMockConn()


# ── upsert_mission ────────────────────────────────────────────────────────────


def test_upsert_mission_creates_record(conn):
    from selfsuvis.pipeline.storage.missions import upsert_mission

    run(
        upsert_mission(
            conn,
            mission_id="m1",
            video_id="v1",
            video_path="/data/v1.mp4",
            job_id="j1",
            robot_id="r1",
            status="indexing",
            frame_count=42,
            duration_sec=120.0,
            gps_origin={"lat": 47.0, "lon": 8.0, "alt": 400.0},
        )
    )

    m = conn._missions["m1"]
    assert m["id"] == "m1"
    assert m["video_id"] == "v1"
    assert m["status"] == "indexing"
    assert m["frame_count"] == 42
    assert m["duration_sec"] == pytest.approx(120.0)
    assert m["gps_origin_json"]["lat"] == pytest.approx(47.0)


def test_upsert_mission_updates_existing(conn):
    from selfsuvis.pipeline.storage.missions import upsert_mission

    run(
        upsert_mission(
            conn,
            mission_id="m1",
            video_id="v1",
            video_path="/v1.mp4",
            job_id="j1",
            robot_id="r1",
            status="indexing",
            frame_count=10,
            duration_sec=30.0,
            gps_origin=None,
        )
    )
    # Second call with same mission_id and updated frame_count
    run(
        upsert_mission(
            conn,
            mission_id="m1",
            video_id="v1",
            video_path="/v1.mp4",
            job_id="j1",
            robot_id="r1",
            status="done",
            frame_count=99,
            duration_sec=30.0,
            gps_origin=None,
        )
    )

    assert conn._missions["m1"]["frame_count"] == 99
    assert conn._missions["m1"]["status"] == "done"


def test_upsert_mission_null_gps_origin(conn):
    from selfsuvis.pipeline.storage.missions import upsert_mission

    run(
        upsert_mission(
            conn,
            mission_id="m2",
            video_id="v2",
            video_path="/v2.mp4",
            job_id="j2",
            robot_id="r1",
            status="indexing",
            frame_count=0,
            duration_sec=None,
            gps_origin=None,
        )
    )

    assert conn._missions["m2"]["gps_origin_json"] is None
    assert conn._missions["m2"]["duration_sec"] is None


# ── replace_frames ────────────────────────────────────────────────────────────


def test_replace_frames_inserts_new_records(conn):
    from selfsuvis.pipeline.storage.missions import replace_frames

    frames = [
        make_frame_record("f1", "m1", t_sec=1.0),
        make_frame_record("f2", "m1", t_sec=2.0),
    ]
    run(replace_frames(conn, "m1", frames))

    assert "f1" in conn._frames
    assert "f2" in conn._frames
    assert conn._frames["f1"]["t_sec"] == pytest.approx(1.0)
    assert conn._frames["f1"]["mission_id"] == "m1"


def test_replace_frames_clears_existing_for_mission(conn):
    from selfsuvis.pipeline.storage.missions import replace_frames

    # Insert initial frames
    run(replace_frames(conn, "m1", [make_frame_record("old1", "m1")]))
    assert "old1" in conn._frames

    # Replace with new frames
    run(replace_frames(conn, "m1", [make_frame_record("new1", "m1")]))
    assert "old1" not in conn._frames
    assert "new1" in conn._frames


def test_replace_frames_does_not_remove_other_missions_frames(conn):
    from selfsuvis.pipeline.storage.missions import replace_frames

    run(replace_frames(conn, "m1", [make_frame_record("f_m1", "m1")]))
    run(replace_frames(conn, "m2", [make_frame_record("f_m2", "m2")]))

    # Replacing m1's frames should leave m2's untouched
    run(replace_frames(conn, "m1", [make_frame_record("f_m1_new", "m1")]))
    assert "f_m2" in conn._frames
    assert "f_m1" not in conn._frames


def test_replace_frames_with_empty_list_clears_mission(conn):
    from selfsuvis.pipeline.storage.missions import replace_frames

    run(replace_frames(conn, "m1", [make_frame_record("f1", "m1")]))
    run(replace_frames(conn, "m1", []))

    m1_frames = [f for f in conn._frames.values() if f["mission_id"] == "m1"]
    assert m1_frames == []


def test_replace_frames_stores_gps_json(conn):
    from selfsuvis.pipeline.storage.missions import replace_frames

    frame = make_frame_record("f1", "m1", gps_lat=47.5, gps_lon=8.5)
    run(replace_frames(conn, "m1", [frame]))

    stored = conn._frames["f1"]
    assert stored["gps_json"]["lat"] == pytest.approx(47.5)
    assert stored["gps_json"]["lon"] == pytest.approx(8.5)


def test_replace_frames_stores_al_tag(conn):
    from selfsuvis.pipeline.storage.missions import replace_frames

    frame = make_frame_record("f1", "m1", al_tag="needs_annotation", al_score=0.87)
    run(replace_frames(conn, "m1", [frame]))

    assert conn._frames["f1"]["al_tag"] == "needs_annotation"
    assert conn._frames["f1"]["al_score"] == pytest.approx(0.87)


# ── mark_mission_finished ─────────────────────────────────────────────────────


def test_mark_mission_finished_sets_done_status(conn):
    from selfsuvis.pipeline.storage.missions import mark_mission_finished, upsert_mission

    run(
        upsert_mission(
            conn,
            mission_id="m1",
            video_id="v1",
            video_path="/v.mp4",
            job_id="j1",
            robot_id="r1",
            status="indexing",
            frame_count=5,
            duration_sec=10.0,
            gps_origin=None,
        )
    )
    run(mark_mission_finished(conn, "m1", status="done"))

    assert conn._missions["m1"]["status"] == "done"


def test_mark_mission_finished_with_error_message(conn):
    from selfsuvis.pipeline.storage.missions import mark_mission_finished, upsert_mission

    run(
        upsert_mission(
            conn,
            mission_id="m1",
            video_id="v1",
            video_path="/v.mp4",
            job_id="j1",
            robot_id="r1",
            status="indexing",
            frame_count=0,
            duration_sec=None,
            gps_origin=None,
        )
    )
    run(mark_mission_finished(conn, "m1", status="error", error="ffmpeg crashed"))

    m = conn._missions["m1"]
    assert m["status"] == "error"
    assert m["error"] == "ffmpeg crashed"


def test_mark_mission_finished_sets_pose_status(conn):
    from selfsuvis.pipeline.storage.missions import mark_mission_finished, upsert_mission

    run(
        upsert_mission(
            conn,
            mission_id="m1",
            video_id="v1",
            video_path="/v.mp4",
            job_id="j1",
            robot_id="r1",
            status="indexing",
            frame_count=3,
            duration_sec=5.0,
            gps_origin=None,
        )
    )
    run(mark_mission_finished(conn, "m1", status="done", pose_status="success"))

    assert conn._missions["m1"]["pose_status"] == "success"


# ── apply_gps_registration ────────────────────────────────────────────────────


def test_apply_gps_registration_updates_enu_origin(conn):
    from selfsuvis.pipeline.storage.missions import apply_gps_registration, upsert_mission

    run(
        upsert_mission(
            conn,
            mission_id="m1",
            video_id="v1",
            video_path="/v.mp4",
            job_id="j1",
            robot_id="r1",
            status="done",
            frame_count=2,
            duration_sec=5.0,
            gps_origin=None,
        )
    )
    run(
        apply_gps_registration(
            conn, "m1", enu_origin={"lat": 47.0, "lon": 8.0, "alt": 400.0}, global_poses={}
        )
    )

    assert conn._missions["m1"]["gps_origin_json"]["lat"] == pytest.approx(47.0)


def test_apply_gps_registration_propagates_frame_poses(conn):
    from selfsuvis.pipeline.storage.missions import apply_gps_registration, replace_frames

    run(
        replace_frames(
            conn,
            "m1",
            [
                make_frame_record("f1", "m1"),
                make_frame_record("f2", "m1"),
            ],
        )
    )
    global_poses = {
        "f1": {"tx": 1.0, "ty": 0.0, "tz": 0.0},
        "f2": {"tx": 2.0, "ty": 0.0, "tz": 0.0},
    }
    run(apply_gps_registration(conn, "m1", enu_origin=None, global_poses=global_poses))

    assert conn._frames["f1"]["global_pose_json"]["tx"] == pytest.approx(1.0)
    assert conn._frames["f2"]["global_pose_json"]["tx"] == pytest.approx(2.0)


# ── list_frames_after ─────────────────────────────────────────────────────────


def test_list_frames_after_no_cursor_returns_first_page(conn):
    from selfsuvis.pipeline.storage.missions import list_frames_after, replace_frames

    frames = [make_frame_record(f"f{i}", "m1", t_sec=float(i)) for i in range(5)]
    run(replace_frames(conn, "m1", frames))

    page = run(list_frames_after(conn, cursor=None, limit=3))
    assert len(page) == 3


def test_list_frames_after_returns_all_when_limit_large(conn):
    from selfsuvis.pipeline.storage.missions import list_frames_after, replace_frames

    frames = [make_frame_record(f"f{i}", "m1") for i in range(4)]
    run(replace_frames(conn, "m1", frames))

    page = run(list_frames_after(conn, cursor=None, limit=100))
    assert len(page) == 4


def test_list_frames_after_cursor_paginates(conn):
    from selfsuvis.pipeline.core.utils import utcnow
    from selfsuvis.pipeline.storage.missions import list_frames_after, replace_frames

    now = utcnow()
    frames = [make_frame_record(f"f{i}", "m1") for i in range(6)]
    run(replace_frames(conn, "m1", frames))

    # Get first page
    first = run(list_frames_after(conn, cursor=None, limit=3))
    assert len(first) == 3

    # Use cursor from last item in first page
    last = first[-1]
    cursor = (last.get("created_at", now), last["id"])
    second = run(list_frames_after(conn, cursor=cursor, limit=3))
    assert len(second) == 3

    # No overlap
    first_ids = {r["id"] for r in first}
    second_ids = {r["id"] for r in second}
    assert first_ids.isdisjoint(second_ids)


def test_list_frames_after_empty_table_returns_empty(conn):
    from selfsuvis.pipeline.storage.missions import list_frames_after

    result = run(list_frames_after(conn, cursor=None, limit=10))
    assert result == []


# ── Full mission lifecycle ────────────────────────────────────────────────────


def test_full_mission_lifecycle(conn):
    """pending → indexing → done, with frames and GPS registration."""
    from selfsuvis.pipeline.storage.jobs import create_job, fetch_and_claim_next_pending, update_job
    from selfsuvis.pipeline.storage.missions import (
        apply_gps_registration,
        list_frames_after,
        mark_mission_finished,
        replace_frames,
        upsert_mission,
    )

    # 1. Create job
    run(create_job(conn, "j1", {"video_id": "v1", "mission_id": "m1"}))
    claimed = run(fetch_and_claim_next_pending(conn))
    assert claimed["id"] == "j1"
    assert claimed["status"] == "running"

    # 2. Upsert mission to "indexing"
    run(
        upsert_mission(
            conn,
            mission_id="m1",
            video_id="v1",
            video_path="/v.mp4",
            job_id="j1",
            robot_id="robot-a",
            status="indexing",
            frame_count=3,
            duration_sec=30.0,
            gps_origin={"lat": 47.0, "lon": 8.0, "alt": 400.0},
        )
    )

    # 3. Persist frames
    frames = [
        make_frame_record(f"f{i}", "m1", t_sec=float(i * 5), gps_lat=47.0, gps_lon=8.0)
        for i in range(3)
    ]
    run(replace_frames(conn, "m1", frames))
    assert len([f for f in conn._frames.values() if f["mission_id"] == "m1"]) == 3

    # 4. GPS registration
    global_poses = {f"f{i}": {"tx": float(i), "ty": 0.0, "tz": 0.0} for i in range(3)}
    run(apply_gps_registration(conn, "m1", enu_origin=None, global_poses=global_poses))
    assert conn._frames["f0"]["global_pose_json"]["tx"] == pytest.approx(0.0)

    # 5. Mark mission done
    run(mark_mission_finished(conn, "m1", status="done", pose_status="success"))
    assert conn._missions["m1"]["status"] == "done"

    # 6. Finalize job
    run(update_job(conn, "j1", status="finished", finished_at=9999.0, progress={"frames": 3}))
    job = run(conn.fetchrow("SELECT * FROM jobs WHERE id = $1", "j1"))
    assert job["status"] == "finished"

    # 7. Frames are pageable
    page = run(list_frames_after(conn, cursor=None, limit=10))
    assert len(page) == 3
