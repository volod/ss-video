"""Mission entry for 4D keyframes and prompted tracks.

``workflows/analysis4d_profile.py`` schedules the fast causal pass and the
deep revision. This module still runs one track pass when a caller asks for
tracks directly.
"""

from datetime import UTC
from pathlib import Path

from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal
from selfsuvis.pipeline.analysis4d.passes import write_profiles
from selfsuvis.pipeline.analysis4d.paths import analysis_dir
from selfsuvis.pipeline.analysis4d.pin import (
    active_count,
    active_grounding,
    active_mask,
    selector_config,
)
from selfsuvis.pipeline.analysis4d.providers import (
    GroundingDinoProvider,
    KinematicMask,
    Prompt,
    ScriptedCount,
    ScriptedGrounding,
    UnavailableGrounding,
)
from selfsuvis.pipeline.analysis4d.schemas import CoordinateFrame
from selfsuvis.pipeline.analysis4d.tracks import TrackerConfig


def run_mission_tracks(
    mission_id: str,
    frames: list[FrameSignal],
    prompts: list[Prompt],
    *,
    dest: Path | None = None,
    grounding: object | None = None,
    counter: object | None = None,
    created_at: str | None = None,
) -> Path:
    """Write fast and deep tracks for one mission.

    Args:
        mission_id: Single path segment under ``$DATA_DIR/analysis``.
        frames: Decoded frames, including embeddings and optional images.
        prompts: Text or exemplar prompts.
        dest: Override for the artifact directory. The default is the mission 4D dir.
        grounding: Provider. The default is the pinned provider.
        counter: Count expert. The default follows ``ANALYSIS4D_COUNT_PROVIDER`` and
            stays off when that variable is empty.
        created_at: Manifest timestamp. The default is the current UTC time.

    Returns:
        The directory that contains ``tracks.jsonl``.
    """
    target = Path(dest) if dest is not None else analysis_dir(mission_id)
    provider = grounding if grounding is not None else _load_grounding(active_grounding())
    count_provider = counter if counter is not None else _load_count(active_count())
    stamp = created_at or _utc_now()
    seed = selector_config()
    return write_profiles(
        mission_id,
        frames,
        prompts,
        target,
        grounding=provider,
        counter=count_provider,
        masker=_load_mask(active_mask()),
        selector=seed,
        tracker_config=TrackerConfig(seed=seed.seed),
        created_at=stamp,
        coordinate_frame=CoordinateFrame(name="mission_enu", metric_scale="unavailable"),
    )


def _load_grounding(name: str):
    if name in {"", "unavailable"}:
        return UnavailableGrounding()
    if name == "grounding_dino":
        return GroundingDinoProvider()
    if name == "scripted":
        return ScriptedGrounding({})
    return UnavailableGrounding()


def _load_mask(name: str):
    del name
    return KinematicMask()


def _load_count(name: str):
    if not name or name == "off":
        return None
    if name == "scripted_count":
        return ScriptedCount({})
    return None


def _utc_now() -> str:
    from datetime import datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
