"""Grep gate: each migration module names only its owner's tables."""

import re
from pathlib import Path

import selfsuvis.fusion_rt.migrate as fusion_migrate_mod

REPO = Path(__file__).resolve().parents[4]
VIDEO_MIGRATE = REPO / "src" / "selfsuvis" / "pipeline" / "storage" / "migrate_video.py"
FUSION_MIGRATE = Path(fusion_migrate_mod.__file__)

FUSION_TABLES = (
    "sensor_keys",
    "site_events",
    "zones",
    "fusion_rules",
    "incidents",
    "incident_notes",
)
VIDEO_TABLES = (
    "jobs",
    "processed_files",
    "missions",
    "frames",
    "embedding_clusters",
    "change_detections",
    "global_map",
    "global_map_missions",
    "cvat_tasks",
    "system_state",
    "model_checkpoints",
    "gpu_jobs",
    "robot_sessions",
    "sensor_packets",
    "realtime_poses",
    "realtime_frames",
    "map_tiles",
    "semantic_observations",
    "scene_timeline",
)


def _named(table: str, text: str) -> bool:
    return re.search(rf"\b{table}\b", text) is not None


def test_video_migration_does_not_name_fusion_tables() -> None:
    text = VIDEO_MIGRATE.read_text(encoding="utf-8")
    found = [table for table in FUSION_TABLES if _named(table, text)]
    assert found == []


def test_fusion_migration_does_not_name_video_tables() -> None:
    text = FUSION_MIGRATE.read_text(encoding="utf-8")
    found = [table for table in VIDEO_TABLES if _named(table, text)]
    assert found == []


def test_each_owner_migration_names_its_tables() -> None:
    video = VIDEO_MIGRATE.read_text(encoding="utf-8")
    fusion = FUSION_MIGRATE.read_text(encoding="utf-8")
    missing_video = [table for table in VIDEO_TABLES if not _named(table, video)]
    missing_fusion = [table for table in FUSION_TABLES if not _named(table, fusion)]
    assert missing_video == []
    assert missing_fusion == []
