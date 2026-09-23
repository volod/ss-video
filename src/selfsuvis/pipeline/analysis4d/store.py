"""Append-only writers for a mission's 4D artifact directory.

JSONL logs reject a repeated id. ``timeline.json`` and ``manifest.json`` are
replaced only when the new document names the previous file's digest.
"""

from pathlib import Path

from selfsuvis.pipeline.analysis4d.io import (
    canonical_bytes,
    canonical_line,
    sha256_bytes,
    write_bytes,
)
from selfsuvis.pipeline.analysis4d.schemas import (
    AnalysisManifest,
    GapRecord,
    GraphDelta,
    Proposal,
    QaRecord,
    SceneTimeline,
    TrackRecord,
)


class AppendOnlyError(ValueError):
    """A write would overwrite history without a supersession link."""


class AnalysisStore:
    """Persist 4D artifacts under one mission directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append_track(self, record: TrackRecord) -> None:
        """Append one track observation."""
        self._append_jsonl("tracks.jsonl", record, record.observation_id)

    def append_delta(self, record: GraphDelta) -> None:
        """Append one graph delta."""
        self._append_jsonl("graph-deltas.jsonl", record, record.delta_id)

    def append_proposal(self, record: Proposal) -> None:
        """Append one proposal."""
        self._append_jsonl("proposals.jsonl", record, record.proposal_id)

    def append_qa(self, record: QaRecord) -> None:
        """Append one QA record."""
        self._append_jsonl("qa.jsonl", record, record.qa_id)

    def append_gap(self, record: GapRecord) -> None:
        """Append one gap record."""
        self._append_jsonl("gaps.jsonl", record, record.gap_id)

    def write_timeline(self, timeline: SceneTimeline) -> str:
        """Write ``timeline.json``, keeping the previous file when it is superseded.

        Args:
            timeline: Materialized timeline. A replacement must set
                ``supersedes_sha256`` to the digest of the current file.

        Returns:
            ``sha256:<hex>`` of the written file.
        """
        return self._replace_document("timeline.json", timeline, timeline.supersedes_sha256)

    def write_manifest(self, manifest: AnalysisManifest) -> str:
        """Write ``manifest.json`` using the same supersession rule as the timeline.

        Returns:
            ``sha256:<hex>`` of the written file.
        """
        return self._replace_document("manifest.json", manifest, manifest.supersedes_sha256)

    def _append_jsonl(self, name: str, record, record_id: str) -> None:
        path = self.root / name
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        marker = f'"{_id_field(name)}":"{record_id}"'
        if marker in existing.replace(" ", ""):
            raise AppendOnlyError(f"{name} already contains {record_id}")
        payload = canonical_line(record)
        with path.open("ab") as handle:
            handle.write(payload)

    def _replace_document(self, name: str, model, supersedes_sha256: str | None) -> str:
        path = self.root / name
        payload = canonical_bytes(model)
        digest = sha256_bytes(payload)
        if not path.is_file():
            if supersedes_sha256 is not None:
                raise AppendOnlyError(f"{name} supersedes a file that does not exist")
            write_bytes(path, payload)
            return digest
        previous = path.read_bytes()
        previous_digest = sha256_bytes(previous)
        if previous == payload:
            return digest
        if supersedes_sha256 != previous_digest:
            raise AppendOnlyError(
                f"{name} already exists; set supersedes_sha256 to {previous_digest}"
            )
        history = self.root / "history"
        history.mkdir(parents=True, exist_ok=True)
        stem = name.replace(".json", "")
        (history / f"{stem}-{previous_digest.removeprefix('sha256:')[:12]}.json").write_bytes(
            previous
        )
        path.write_bytes(payload)
        return digest


def _id_field(name: str) -> str:
    return {
        "tracks.jsonl": "observation_id",
        "graph-deltas.jsonl": "delta_id",
        "proposals.jsonl": "proposal_id",
        "qa.jsonl": "qa_id",
        "gaps.jsonl": "gap_id",
    }[name]
