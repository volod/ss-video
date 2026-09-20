"""Shared in-process database helpers for tests."""

import json
from datetime import datetime, timezone
from typing import Any


class Row(dict):
    """dict subclass that supports asyncpg row.column access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class PipelineMockConn:
    """Simulate asyncpg Connection behavior used by pipeline integration tests."""

    def __init__(self):
        self._jobs: dict[str, Row] = {}
        self._missions: dict[str, Row] = {}
        self._frames: dict[str, Row] = {}
        self._timeline: list[Row] = []
        self._processed: dict[str, Row] = {}

    @staticmethod
    def _min_dt() -> datetime:
        return datetime.min.replace(tzinfo=timezone.utc)

    async def execute(self, query: str, *args) -> str:
        q = query.strip().upper()

        if "INSERT INTO JOBS" in q:
            job_id, status, job_type, prog_j, payload_j, created_at = args
            prog = json.loads(prog_j) if isinstance(prog_j, str) else (prog_j or {})
            payload = json.loads(payload_j) if isinstance(payload_j, str) else (payload_j or {})
            self._jobs[job_id] = Row(
                id=job_id,
                status=status,
                type=job_type,
                progress=prog,
                progress_json=prog,
                payload=payload,
                payload_json=payload,
                created_at=created_at,
                started_at=None,
                finished_at=None,
                error=None,
            )
        elif "UPDATE JOBS SET STATUS" in q and "'RUNNING'" in q:
            started_at, job_id = args
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["started_at"] = started_at
        elif "UPDATE JOBS SET" in q:
            *vals, job_id = args
            if job_id not in self._jobs:
                return "UPDATE 0"
            set_part = query[query.upper().index("SET") + 3 : query.upper().index("WHERE")].strip()
            cols = [p.split("=")[0].strip() for p in set_part.split(",")]
            for col, val in zip(cols, vals):
                if col in ("progress_json", "progress"):
                    val = json.loads(val) if isinstance(val, str) else val
                    self._jobs[job_id]["progress_json"] = val
                    self._jobs[job_id]["progress"] = val
                elif col in ("payload_json", "payload"):
                    val = json.loads(val) if isinstance(val, str) else val
                    self._jobs[job_id]["payload_json"] = val
                    self._jobs[job_id]["payload"] = val
                self._jobs[job_id][col.replace("_json", "")] = val

        elif "INSERT INTO MISSIONS" in q:
            (
                m_id,
                vid_id,
                vid_path,
                job_id,
                robot_id,
                status,
                pose_st,
                map_st,
                frame_cnt,
                dur,
                gps_j,
                created_at,
                updated_at,
                error,
            ) = args
            gps = json.loads(gps_j) if gps_j else None
            self._missions[m_id] = Row(
                id=m_id,
                video_id=vid_id,
                video_path=vid_path,
                job_id=job_id,
                robot_id=robot_id,
                status=status,
                pose_status=pose_st,
                map_status=map_st,
                frame_count=frame_cnt,
                duration_sec=dur,
                gps_origin_json=gps,
                error=error,
            )
        elif "UPDATE MISSIONS SET" in q:
            if "GPS_ORIGIN_JSON" in q:
                gps_j, updated_at, m_id = args
                if m_id in self._missions:
                    self._missions[m_id]["gps_origin_json"] = json.loads(gps_j)
            else:
                *vals, m_id = args
                if m_id not in self._missions:
                    return "UPDATE 0"
                set_part = query[
                    query.upper().index("SET") + 3 : query.upper().index("WHERE")
                ].strip()
                cols = [p.split("=")[0].strip() for p in set_part.split(",")]
                for col, val in zip(cols, vals):
                    self._missions[m_id][col] = val

        elif "DELETE FROM FRAMES" in q:
            mission_id = args[0]
            self._frames = {k: v for k, v in self._frames.items() if v["mission_id"] != mission_id}
        elif "UPDATE FRAMES SET" in q and "GLOBAL_POSE_JSON" in q:
            pass

        elif "INSERT INTO SCENE_TIMELINE" in q:
            (m_id, f_id, lat, lon, alt, t_sec, cap, facts_j, created_at) = args
            facts = json.loads(facts_j) if isinstance(facts_j, str) else facts_j
            self._timeline.append(
                Row(
                    mission_id=m_id,
                    frame_id=f_id,
                    gps_lat=lat,
                    gps_lon=lon,
                    gps_alt=alt,
                    t_sec=t_sec,
                    caption=cap,
                    facts_json=facts,
                    created_at=created_at,
                )
            )

        return "OK"

    async def executemany(self, query: str, rows) -> None:
        q = query.strip().upper()
        if "INSERT INTO FRAMES" in q:
            for row in rows:
                (
                    f_id,
                    m_id,
                    fpath,
                    t_sec,
                    seg_id,
                    cap,
                    cap_conf,
                    cap_model,
                    sub,
                    ocr,
                    facts_j,
                    al_score,
                    al_tag,
                    cvat,
                    pose_st,
                    pose_j,
                    gps_j,
                    gpj,
                    qid,
                    created_at,
                    updated_at,
                ) = row
                self._frames[f_id] = Row(
                    id=f_id,
                    mission_id=m_id,
                    frame_path=fpath,
                    t_sec=t_sec,
                    segment_id=seg_id,
                    caption=cap,
                    caption_confidence=cap_conf,
                    caption_model=cap_model,
                    al_tag=al_tag,
                    al_score=al_score,
                    gps_json=json.loads(gps_j) if gps_j else None,
                    global_pose_json=json.loads(gpj) if gpj else None,
                    qdrant_id=qid,
                    created_at=created_at,
                    updated_at=updated_at,
                )
        elif "UPDATE FRAMES" in q and "GLOBAL_POSE_JSON" in q:
            for pose_j, updated_at, f_id in rows:
                if f_id in self._frames:
                    self._frames[f_id]["global_pose_json"] = json.loads(pose_j)

    async def fetchrow(self, query: str, *args) -> Row | None:
        q = query.strip().upper()
        if "FROM JOBS WHERE ID" in q:
            return self._jobs.get(args[0])
        if "FROM JOBS" in q and "STATUS = 'PENDING'" in q:
            pending = [j for j in self._jobs.values() if j["status"] == "pending"]
            return min(pending, key=lambda j: j["created_at"]) if pending else None
        if "FROM MISSIONS WHERE ID" in q:
            return self._missions.get(args[0])
        return None

    async def fetchval(self, query: str, *args) -> Any:
        q = query.strip().upper()
        if "COUNT(*)" in q and "JOBS" in q:
            return sum(1 for j in self._jobs.values() if j["status"] == "pending")
        if "COUNT(*)" in q and "MISSIONS" in q:
            return (
                sum(1 for m in self._missions.values() if m["status"] == args[0])
                if args
                else len(self._missions)
            )
        return None

    async def fetch(self, query: str, *args) -> list[Row]:
        q = query.strip().upper()

        if "FROM FRAMES" in q and "ORDER BY CREATED_AT ASC" in q:
            sorted_frames = sorted(
                self._frames.values(),
                key=lambda f: (f.get("created_at") or self._min_dt(), f["id"]),
            )
            limit = args[-1] if args else 100
            if "(CREATED_AT, ID) >" in q:
                cursor_ts, cursor_id, limit = args
                sorted_frames = [
                    f
                    for f in sorted_frames
                    if (f.get("created_at") or self._min_dt(), f["id"]) > (cursor_ts, cursor_id)
                ]
            return sorted_frames[:limit]

        if "FROM SCENE_TIMELINE" in q:
            min_lat, max_lat, min_lon, max_lon, limit = args
            nearby = [
                t
                for t in self._timeline
                if (
                    t.get("gps_lat") is not None
                    and min_lat <= t["gps_lat"] <= max_lat
                    and min_lon <= t["gps_lon"] <= max_lon
                )
            ]
            nearby.sort(key=lambda r: r.get("created_at") or self._min_dt(), reverse=True)
            return nearby[:limit]

        return []

    class _TxCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    def transaction(self):
        return self._TxCtx()


class PipelineMockPool:
    """Wrap PipelineMockConn so it supports `async with pool.acquire() as conn`."""

    def __init__(self, conn: PipelineMockConn | None = None):
        self.conn = conn or PipelineMockConn()

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self_inner):
                return pool.conn

            async def __aexit__(self_inner, *_):
                pass

        return _Ctx()

    async def execute(self, *args, **kwargs):
        return await self.conn.execute(*args, **kwargs)

    async def fetch(self, *args, **kwargs):
        return await self.conn.fetch(*args, **kwargs)

    async def fetchrow(self, *args, **kwargs):
        return await self.conn.fetchrow(*args, **kwargs)

    async def fetchval(self, *args, **kwargs):
        return await self.conn.fetchval(*args, **kwargs)


def make_frame_record(
    frame_id: str = "f001",
    mission_id: str = "m1",
    frame_path: str = "/data/frames/f001.jpg",
    t_sec: float = 5.0,
    gps_lat: float = 47.0,
    gps_lon: float = 8.0,
    al_tag: str = "none",
    al_score: float = 0.1,
    caption: str | None = None,
    qdrant_id: int | None = None,
) -> dict[str, Any]:
    return {
        "id": frame_id,
        "mission_id": mission_id,
        "frame_path": frame_path,
        "t_sec": t_sec,
        "segment_id": None,
        "caption": caption,
        "caption_confidence": None,
        "caption_model": None,
        "subtitle_text": None,
        "ocr_text": None,
        "frame_facts_json": None,
        "al_score": al_score,
        "al_tag": al_tag,
        "cvat_label": None,
        "pose_status": "pending",
        "pose_json": None,
        "gps_json": {"lat": gps_lat, "lon": gps_lon},
        "global_pose_json": None,
        "qdrant_id": qdrant_id,
    }
