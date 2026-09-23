"""Query metadata rows and the video-database insert/list helpers."""

import asyncio
from pathlib import Path

from selfsuvis.pipeline.analysis4d.io import canonical_bytes
from selfsuvis.pipeline.analysis4d.metadata import metadata_rows
from selfsuvis.pipeline.analysis4d.validate import validate_bundle
from selfsuvis.pipeline.storage.analysis4d import (
    list_edges,
    list_events,
    list_qa,
    replace_analysis_run,
)

ASSETS = Path(__file__).resolve().parents[4] / "tests" / "assets" / "analysis4d"


class _Conn:
    """Just enough of asyncpg to record one run and answer the list queries."""

    def __init__(self):
        self.events: list[dict] = []
        self.edges: list[dict] = []
        self.qa: list[dict] = []
        self.runs: list[dict] = []

    async def execute(self, query: str, *args) -> None:
        text = " ".join(query.split())
        if text.startswith("DELETE"):
            return
        if "INSERT INTO analysis4d_runs" in text:
            self.runs.append({"id": args[0], "mission_id": args[1]})
        elif "INSERT INTO analysis4d_events" in text:
            self.events.append(
                {
                    "mission_id": args[2],
                    "event_id": args[1],
                    "start_sec": args[5],
                    "verification_status": args[7],
                }
            )
        elif "INSERT INTO analysis4d_edges" in text:
            self.edges.append(
                {
                    "mission_id": args[2],
                    "edge_id": args[1],
                    "predicate": args[4],
                    "start_sec": args[6],
                }
            )
        elif "INSERT INTO analysis4d_qa" in text:
            self.qa.append(
                {
                    "mission_id": args[2],
                    "qa_id": args[1],
                    "interval_start_sec": args[8],
                }
            )

    async def fetch(self, query: str, *args):
        text = " ".join(query.split())
        if "analysis4d_events" in text:
            rows = [row for row in self.events if row["mission_id"] == args[0]]
            if "verification_status = $2" in text:
                rows = [row for row in rows if row["verification_status"] == args[1]]
            return rows
        if "analysis4d_edges" in text:
            rows = [row for row in self.edges if row["mission_id"] == args[0]]
            if len(args) == 2:
                rows = [row for row in rows if row["predicate"] == args[1]]
            return rows
        return [row for row in self.qa if row["mission_id"] == args[0]]


def test_metadata_round_trip() -> None:
    bundle = validate_bundle(ASSETS / "nominal", require_truth=True)
    manifest_bytes = (ASSETS / "nominal" / "manifest.json").read_bytes()
    assert manifest_bytes == canonical_bytes(bundle.manifest)
    rows = metadata_rows(
        bundle, artifact_dir="analysis/mission-nominal/4d", manifest_bytes=manifest_bytes
    )
    assert rows["events"][0]["verification_status"] == "accepted"
    assert rows["edges"][0]["predicate"] == "contains"
    assert rows["qa"][0]["answer_value"] == "track-1"
    conn = _Conn()
    asyncio.run(replace_analysis_run(conn, rows))
    events = asyncio.run(list_events(conn, "mission-nominal", status="accepted"))
    edges = asyncio.run(list_edges(conn, "mission-nominal", predicate="contains"))
    qa = asyncio.run(list_qa(conn, "mission-nominal"))
    assert [row["event_id"] for row in events] == ["evt-1"]
    assert [row["edge_id"] for row in edges] == ["edge-contains"]
    assert [row["qa_id"] for row in qa] == ["qa-1"]
