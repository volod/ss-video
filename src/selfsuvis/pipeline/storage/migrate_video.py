"""Video-owned PostgreSQL schema (jobs, missions, frames, robot realtime)."""

import asyncpg

from selfsuvis.pipeline.core.ddl import apply_statements, with_advisory_lock

VIDEO_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id            TEXT PRIMARY KEY,
        status        TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending','running','finished','error')),
        type          TEXT,
        progress_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        payload_json  JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        started_at    TIMESTAMPTZ,
        finished_at   TIMESTAMPTZ,
        error         TEXT
    )

    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_type_status ON jobs (type, status)",
    """
    CREATE TABLE IF NOT EXISTS processed_files (
        file_hash   TEXT PRIMARY KEY,
        video_id    TEXT NOT NULL,
        path        TEXT,
        size_bytes  BIGINT,
        mtime       DOUBLE PRECISION,
        status      TEXT NOT NULL DEFAULT 'done',
        meta_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_processed_files_size_updated ON processed_files (size_bytes, updated_at DESC)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_processed_files_url ON processed_files ((meta_json ->> 'url'))
    """,
    """
    CREATE TABLE IF NOT EXISTS missions (
        id              TEXT PRIMARY KEY,
        video_id        TEXT NOT NULL,
        video_path      TEXT,
        job_id          TEXT REFERENCES jobs(id),
        robot_id        TEXT NOT NULL DEFAULT 'robot_0',
        status          TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending','indexing','done','error')),
        pose_status     TEXT NOT NULL DEFAULT 'pending'
                            CHECK (pose_status IN ('pending','running','success','failed','skipped')),
        map_status      TEXT NOT NULL DEFAULT 'pending'
                            CHECK (map_status IN ('pending','running','success','failed','skipped')),
        frame_count     INTEGER NOT NULL DEFAULT 0,
        duration_sec    DOUBLE PRECISION,
        gps_origin_json JSONB,
        splat_path      TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        error           TEXT
    )

    """,
    "CREATE INDEX IF NOT EXISTS idx_missions_status ON missions (status)",
    "CREATE INDEX IF NOT EXISTS idx_missions_video_id ON missions (video_id)",
    "CREATE INDEX IF NOT EXISTS idx_missions_robot_id ON missions (robot_id)",
    "CREATE INDEX IF NOT EXISTS idx_missions_created_at ON missions (created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS frames (
        id                  TEXT PRIMARY KEY,
        mission_id          TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
        frame_path          TEXT NOT NULL,
        t_sec               DOUBLE PRECISION NOT NULL,
        segment_id          INTEGER,
        caption             TEXT,
        caption_confidence  DOUBLE PRECISION,
        caption_model       TEXT,
        caption_skip_reason TEXT,
        subtitle_text       TEXT,
        ocr_text            TEXT,
        frame_facts_json    JSONB,
        al_score            DOUBLE PRECISION,
        al_tag              TEXT NOT NULL DEFAULT 'none'
                                CHECK (al_tag IN ('none','needs_annotation','novel','annotated')),
        cvat_label          TEXT,
        pose_status         TEXT NOT NULL DEFAULT 'pending'
                                CHECK (pose_status IN ('pending','success','failed','skipped')),
        pose_json           JSONB,
        gps_json            JSONB,
        global_pose_json    JSONB,
        qdrant_id           BIGINT,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    "CREATE INDEX IF NOT EXISTS idx_frames_mission_id ON frames (mission_id)",
    "CREATE INDEX IF NOT EXISTS idx_frames_al_tag ON frames (al_tag)",
    "CREATE INDEX IF NOT EXISTS idx_frames_pose_status ON frames (pose_status)",
    "CREATE INDEX IF NOT EXISTS idx_frames_t_sec ON frames (t_sec)",
    """
CREATE INDEX IF NOT EXISTS idx_frames_al_tag_score ON frames (al_tag, al_score DESC NULLS LAST)
    """,
    "CREATE INDEX IF NOT EXISTS idx_frames_created_id ON frames (created_at, id)",
    """
CREATE INDEX IF NOT EXISTS idx_frames_annotated_label ON frames (mission_id, cvat_label) WHERE al_tag = 'annotated' AND cvat_label IS NOT NULL
    """,
    """
CREATE INDEX IF NOT EXISTS idx_frames_caption_fts ON frames USING gin(to_tsvector('english', coalesce(caption, '')))
    """,
    """
CREATE INDEX IF NOT EXISTS idx_frames_facts_gin ON frames USING gin(frame_facts_json) WHERE frame_facts_json IS NOT NULL
    """,
    """
CREATE INDEX IF NOT EXISTS idx_frames_subtitle_fts ON frames USING gin(to_tsvector('english', coalesce(subtitle_text, ''))) WHERE subtitle_text IS NOT NULL
    """,
    """
CREATE INDEX IF NOT EXISTS idx_frames_ocr_fts ON frames USING gin(to_tsvector('english', coalesce(ocr_text, ''))) WHERE ocr_text IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_clusters (
        id            SERIAL PRIMARY KEY,
        mission_id    TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
        cluster_id    INTEGER NOT NULL,
        centroid_json JSONB NOT NULL,
        frame_count   INTEGER NOT NULL DEFAULT 0,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (mission_id, cluster_id)
    )

    """,
    """
    CREATE TABLE IF NOT EXISTS change_detections (
        id                  SERIAL PRIMARY KEY,
        frame_id            TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
        mission_id          TEXT NOT NULL,
        ref_frame_id        TEXT NOT NULL,
        ref_mission_id      TEXT NOT NULL,
        change_score        DOUBLE PRECISION NOT NULL,
        threshold           DOUBLE PRECISION NOT NULL,
        semantic_diff_json  JSONB,
        change_explanation  TEXT,
        detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    "ALTER TABLE change_detections ADD COLUMN IF NOT EXISTS semantic_diff_json JSONB",
    "ALTER TABLE change_detections ADD COLUMN IF NOT EXISTS change_explanation TEXT",
    """
CREATE INDEX IF NOT EXISTS idx_change_detections_mission_id ON change_detections (mission_id)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_change_detections_score ON change_detections (change_score)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_change_detections_semantic_diff ON change_detections USING gin(semantic_diff_json) WHERE semantic_diff_json IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS global_map (
        id          SERIAL PRIMARY KEY,
        origin_lat  DOUBLE PRECISION NOT NULL,
        origin_lon  DOUBLE PRECISION NOT NULL,
        origin_alt  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        splat_path  TEXT,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_global_map_origin ON global_map (origin_lat, origin_lon)
    """,
    """
    CREATE TABLE IF NOT EXISTS global_map_missions (
        id                            SERIAL PRIMARY KEY,
        global_map_id                 INTEGER NOT NULL REFERENCES global_map(id) ON DELETE CASCADE,
        mission_id                    TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
        registration_transform_json   JSONB NOT NULL,
        registration_error            DOUBLE PRECISION,
        registered_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (global_map_id, mission_id)
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_gmm_global_map_id ON global_map_missions (global_map_id)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_gmm_mission_id ON global_map_missions (mission_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS cvat_tasks (
        cvat_task_id  INTEGER NOT NULL,
        frame_id      TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (cvat_task_id, frame_id)
    )

    """,
    "CREATE INDEX IF NOT EXISTS idx_cvat_tasks_task_id ON cvat_tasks (cvat_task_id)",
    """
    CREATE TABLE IF NOT EXISTS system_state (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
    CREATE TABLE IF NOT EXISTS model_checkpoints (
        id                 SERIAL PRIMARY KEY,
        checkpoint_path    TEXT NOT NULL UNIQUE,
        model_version_id   TEXT NOT NULL,
        annotation_count   INTEGER NOT NULL DEFAULT 0,
        best_accuracy      DOUBLE PRECISION,
        distribution_shift DOUBLE PRECISION,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        notes              TEXT
    )

    """,
    """
    CREATE TABLE IF NOT EXISTS gpu_jobs (
        job_id     TEXT PRIMARY KEY,
        job_type   TEXT NOT NULL,
        worker_id  TEXT NOT NULL,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
    CREATE TABLE IF NOT EXISTS robot_sessions (
        id                  TEXT PRIMARY KEY,
        robot_id            TEXT NOT NULL,
        mission_id          TEXT,
        sensor_profile_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        status              TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active','stopped','failed')),
        started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        ended_at            TIMESTAMPTZ,
        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_robot_sessions_robot_id_started ON robot_sessions (robot_id, started_at DESC)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_robot_sessions_status_started ON robot_sessions (status, started_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS sensor_packets (
        id           BIGSERIAL PRIMARY KEY,
        session_id   TEXT NOT NULL REFERENCES robot_sessions(id) ON DELETE CASCADE,
        sensor_type  TEXT NOT NULL,
        t_device     DOUBLE PRECISION NOT NULL,
        t_server     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        seq          BIGINT,
        payload_json JSONB NOT NULL DEFAULT '{}'::jsonb
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_sensor_packets_session_sensor_time ON sensor_packets (session_id, sensor_type, t_device DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS realtime_poses (
        id                    BIGSERIAL PRIMARY KEY,
        session_id            TEXT NOT NULL REFERENCES robot_sessions(id) ON DELETE CASCADE,
        source                TEXT NOT NULL,
        t_sec                 DOUBLE PRECISION NOT NULL,
        position_enu_json     JSONB NOT NULL,
        orientation_quat_json JSONB,
        velocity_enu_json     JSONB,
        covariance_json       JSONB,
        tracking_status       TEXT NOT NULL DEFAULT 'ok',
        global_map_id         INTEGER REFERENCES global_map(id) ON DELETE SET NULL,
        created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_realtime_poses_session_tsec ON realtime_poses (session_id, t_sec DESC)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_realtime_poses_global_map_time ON realtime_poses (global_map_id, created_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS realtime_frames (
        id         BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES robot_sessions(id) ON DELETE CASCADE,
        frame_id   TEXT NOT NULL,
        t_sec      DOUBLE PRECISION NOT NULL,
        image_path TEXT NOT NULL,
        pose_json  JSONB,
        depth_path TEXT,
        tile_key   TEXT,
        map_type   TEXT NOT NULL DEFAULT 'occupancy',
        stats_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (session_id, frame_id)
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_realtime_frames_session_time ON realtime_frames (session_id, t_sec DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS map_tiles (
        id            BIGSERIAL PRIMARY KEY,
        session_id    TEXT NOT NULL REFERENCES robot_sessions(id) ON DELETE CASCADE,
        global_map_id INTEGER REFERENCES global_map(id) ON DELETE SET NULL,
        tile_key      TEXT NOT NULL,
        map_type      TEXT NOT NULL,
        storage_path  TEXT NOT NULL,
        resolution_m  DOUBLE PRECISION NOT NULL,
        bounds_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
        stats_json    JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (session_id, tile_key, map_type)
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_map_tiles_session_type_updated ON map_tiles (session_id, map_type, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_observations (
        id                BIGSERIAL PRIMARY KEY,
        session_id        TEXT NOT NULL REFERENCES robot_sessions(id) ON DELETE CASCADE,
        frame_id          TEXT,
        class_name        TEXT NOT NULL,
        confidence        DOUBLE PRECISION NOT NULL,
        position_enu_json JSONB,
        bbox_json         JSONB,
        mask_ref          TEXT,
        track_id          TEXT,
        facts_json        JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_semantic_observations_session_created ON semantic_observations (session_id, created_at DESC)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_semantic_observations_class_created ON semantic_observations (class_name, created_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS scene_timeline (
        id          BIGSERIAL PRIMARY KEY,
        mission_id  TEXT NOT NULL,
        frame_id    TEXT NOT NULL,
        gps_lat     DOUBLE PRECISION,
        gps_lon     DOUBLE PRECISION,
        gps_alt     DOUBLE PRECISION,
        t_sec       DOUBLE PRECISION,
        caption     TEXT,
        facts_json  JSONB,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )

    """,
    """
CREATE INDEX IF NOT EXISTS idx_scene_timeline_mission_id ON scene_timeline (mission_id)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_scene_timeline_gps_lat ON scene_timeline (gps_lat)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_scene_timeline_frame_id ON scene_timeline (frame_id)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_scene_timeline_created_at ON scene_timeline (created_at DESC)
    """,
    """
CREATE INDEX IF NOT EXISTS idx_scene_timeline_facts ON scene_timeline USING gin(facts_json) WHERE facts_json IS NOT NULL
    """,
]

VIDEO_SCHEMA_LOCK_KEY = 87263401


async def apply_schema(conn, *, verbose: bool = True) -> None:
    """Apply every video DDL statement to an open connection."""
    await apply_statements(conn, VIDEO_SCHEMA, verbose=verbose)


async def migrate(url: str, *, verbose: bool = True) -> None:
    """Create the video schema. Safe to re-run. Does not create the database."""
    if verbose:
        print(f"Connecting to: {url.split('@')[-1]}")
    conn = await asyncpg.connect(url)
    try:

        async def _body():
            await apply_schema(conn, verbose=verbose)

        await with_advisory_lock(conn, VIDEO_SCHEMA_LOCK_KEY, _body)
        if verbose:
            print(f"\nVideo schema bootstrap complete - {len(VIDEO_SCHEMA)} statements applied.")
    finally:
        await conn.close()
