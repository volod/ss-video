"""4D scene-analysis contracts, artifact store, and fixture benchmark."""

from selfsuvis.pipeline.analysis4d.paths import analysis_dir
from selfsuvis.pipeline.analysis4d.schemas import (
    METRIC_NAMES,
    SCHEMA_BENCHMARK,
    SCHEMA_MANIFEST,
    SCHEMA_TIMELINE,
)
from selfsuvis.pipeline.analysis4d.validate import ContractError, validate_bundle

__all__ = [
    "METRIC_NAMES",
    "SCHEMA_BENCHMARK",
    "SCHEMA_MANIFEST",
    "SCHEMA_TIMELINE",
    "ContractError",
    "analysis_dir",
    "validate_bundle",
]
