"""Shared in-memory asyncpg-style realtime DB fakes for tests."""

import json
from datetime import datetime, timezone
from typing import Any


class AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeRealtimePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return AcquireCtx(self._conn)


class FakeRealtimeConn:
    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}
        self.packets: list[dict[str, Any]] = []
        self.poses: list[dict[str, Any]] = []
        self.tiles: list[dict[str, Any]] = []
        self.semantic: list[dict[str, Any]] = []
        self.realtime_frames: list[dict[str, Any]] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.missions: dict[str, dict[str, Any]] = {}

    async def execute(self, query: str, *args):
        if "INSERT INTO robot_sessions" in query:
            self.sessions[args[0]] = {
                "id": args[0],
                "robot_id": args[1],
                "mission_id": args[2],
                "sensor_profile_json": json.loads(args[3]),
                "status": args[4],
                "started_at": args[5],
                "ended_at": None,
                "updated_at": args[6],
            }
            return "INSERT 0 1"
        if "UPDATE robot_sessions" in query:
            session = self.sessions[args[2]]
            session["status"] = args[0]
            session["ended_at"] = args[1]
            session["updated_at"] = args[1]
            return "UPDATE 1"
        if "INSERT INTO realtime_poses" in query:
            self.poses.append(
                {
                    "id": len(self.poses) + 1,
                    "session_id": args[0],
                    "source": args[1],
                    "t_sec": args[2],
                    "position_enu_json": json.loads(args[3]),
                    "orientation_quat_json": json.loads(args[4]) if args[4] else None,
                    "velocity_enu_json": json.loads(args[5]) if args[5] else None,
                    "covariance_json": json.loads(args[6]) if args[6] else None,
                    "tracking_status": args[7],
                    "global_map_id": args[8],
                    "created_at": datetime.now(timezone.utc),
                }
            )
            return "INSERT 0 1"
        if "INSERT INTO map_tiles" in query:
            row = {
                "session_id": args[0],
                "global_map_id": args[1],
                "tile_key": args[2],
                "map_type": args[3],
                "storage_path": args[4],
                "resolution_m": args[5],
                "bounds_json": json.loads(args[6]),
                "stats_json": json.loads(args[7]),
                "updated_at": args[8],
            }
            self.tiles = [
                existing
                for existing in self.tiles
                if not (
                    existing["session_id"] == row["session_id"]
                    and existing["tile_key"] == row["tile_key"]
                    and existing["map_type"] == row["map_type"]
                )
            ]
            self.tiles.append(row)
            return "INSERT 0 1"
        if "INSERT INTO semantic_observations" in query:
            self.semantic.append(
                {
                    "id": len(self.semantic) + 1,
                    "session_id": args[0],
                    "frame_id": args[1],
                    "class_name": args[2],
                    "confidence": args[3],
                    "position_enu_json": json.loads(args[4]) if args[4] else None,
                    "bbox_json": json.loads(args[5]) if args[5] else None,
                    "mask_ref": args[6],
                    "track_id": args[7],
                    "facts_json": json.loads(args[8]),
                    "created_at": datetime.now(timezone.utc),
                }
            )
            return "INSERT 0 1"
        if "INSERT INTO realtime_frames" in query:
            row = {
                "session_id": args[0],
                "frame_id": args[1],
                "t_sec": args[2],
                "image_path": args[3],
                "pose_json": json.loads(args[4]) if args[4] else None,
                "depth_path": args[5],
                "tile_key": args[6],
                "map_type": args[7],
                "stats_json": json.loads(args[8]),
                "updated_at": args[9],
            }
            self.realtime_frames = [
                existing
                for existing in self.realtime_frames
                if not (
                    existing["session_id"] == row["session_id"]
                    and existing["frame_id"] == row["frame_id"]
                )
            ]
            self.realtime_frames.append(row)
            return "INSERT 0 1"
        if "INSERT INTO jobs" in query:
            self.jobs[args[0]] = {
                "id": args[0],
                "type": args[2],
                "payload_json": json.loads(args[4]),
            }
            return "INSERT 0 1"
        if "INSERT INTO missions" in query:
            self.missions[args[0]] = {
                "id": args[0],
                "video_id": args[1],
                "video_path": args[2],
                "job_id": args[3],
                "robot_id": args[4],
                "status": args[5],
            }
            return "INSERT 0 1"
        raise AssertionError(f"unexpected execute query: {query}")

    async def executemany(self, query: str, rows):
        rows = list(rows)
        if "INSERT INTO sensor_packets" in query:
            for row in rows:
                self.packets.append(
                    {
                        "session_id": row[0],
                        "sensor_type": row[1],
                        "t_device": row[2],
                        "t_server": row[3],
                        "seq": row[4],
                        "payload_json": json.loads(row[5]),
                    }
                )
            return
        raise AssertionError(f"unexpected executemany query: {query}")

    async def fetchrow(self, query: str, *args) -> dict[str, Any] | None:
        if "FROM robot_sessions" in query:
            return self.sessions.get(args[0])
        if "FROM realtime_poses" in query:
            matches = [row for row in self.poses if row["session_id"] == args[0]]
            if not matches:
                return None
            return sorted(matches, key=lambda row: (row["t_sec"], row["id"]), reverse=True)[0]
        if "MIN(t_device) AS t_min" in query:
            session_id = args[0]
            times = [row["t_device"] for row in self.packets if row["session_id"] == session_id]
            if not times:
                return {"t_min": None, "t_max": None}
            return {"t_min": min(times), "t_max": max(times)}
        raise AssertionError(f"unexpected fetchrow query: {query}")

    async def fetch(self, query: str, *args):
        if "FROM sensor_packets" in query and "GROUP BY sensor_type" in query:
            session_id = args[0]
            counts: dict[str, int] = {}
            for row in self.packets:
                if row["session_id"] != session_id:
                    continue
                counts[row["sensor_type"]] = counts.get(row["sensor_type"], 0) + 1
            return [{"sensor_type": key, "n": value} for key, value in sorted(counts.items())]
        if "FROM map_tiles" in query:
            session_id = args[0]
            if len(args) == 3:
                map_type = args[1]
                limit = args[2]
                rows = [
                    row
                    for row in self.tiles
                    if row["session_id"] == session_id and row["map_type"] == map_type
                ]
            else:
                limit = args[1]
                rows = [row for row in self.tiles if row["session_id"] == session_id]
            return list(reversed(rows))[:limit]
        if "FROM semantic_observations" in query:
            session_id = args[0]
            if len(args) == 3:
                class_name = args[1]
                limit = args[2]
                rows = [
                    row
                    for row in self.semantic
                    if row["session_id"] == session_id and row["class_name"] == class_name
                ]
            else:
                limit = args[1]
                rows = [row for row in self.semantic if row["session_id"] == session_id]
            return list(reversed(rows))[:limit]
        if "FROM realtime_frames" in query:
            session_id = args[0]
            limit = args[1]
            rows = [row for row in self.realtime_frames if row["session_id"] == session_id]
            return rows[:limit]
        raise AssertionError(f"unexpected fetch query: {query}")
