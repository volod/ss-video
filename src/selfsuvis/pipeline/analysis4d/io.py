"""Canonical JSON and append-only artifact IO."""

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from selfsuvis.pipeline.analysis4d.schemas import ContractModel

_ModelT = TypeVar("_ModelT", bound=ContractModel)


def canonical_bytes(model: BaseModel) -> bytes:
    """Serialize a model with sorted keys and a trailing newline."""
    payload = model.model_dump(mode="json")
    text = json.dumps(payload, indent=2, sort_keys=True)
    return (text + "\n").encode("utf-8")


def canonical_line(model: BaseModel) -> bytes:
    """Serialize one JSONL record, sorted keys, no extra whitespace."""
    payload = model.model_dump(mode="json")
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return ``sha256:<hex>`` for ``payload``."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    """Create parents and write ``payload``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def read_model(path: Path, model_type: type[_ModelT]) -> _ModelT:
    """Load one JSON document."""
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, model_type: type[_ModelT]) -> list[_ModelT]:
    """Load a JSONL file. A missing file is an empty log. Blank lines are skipped."""
    if not path.is_file():
        return []
    rows: list[_ModelT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(model_type.model_validate_json(text))
        except ValidationError as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
    return rows


def load_json_object(path: Path) -> dict:
    """Load a JSON object, or raise ``ValueError`` when the top level is not an object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value
