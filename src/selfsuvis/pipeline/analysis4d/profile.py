"""Profile selection for 4D analysis.

``off`` is the default. Indexing then follows the current video-search path
and does not enqueue a 4D job. ``fast`` adds the causal job. ``deep`` adds
that job and a postflight revision that supersedes it.
"""

import os

OFF = "off"
FAST = "fast"
DEEP = "deep"
PROFILES = frozenset({OFF, FAST, DEEP})
FAST_JOB = "analysis4d_fast"
DEEP_JOB = "postflight_analysis4d_deep"
_FAST_TYPES = "entered_region,left_region,count_changed"


def normalize_profile(raw: str | None) -> str:
    """Return ``off``, ``fast``, or ``deep``. Anything else is ``off``."""
    value = (raw or "").strip().lower()
    return value if value in PROFILES else OFF


def configured_profile() -> str:
    """Return the process profile. The default is ``off``."""
    return normalize_profile(os.environ.get("ANALYSIS4D_PROFILE", OFF))


def stamp_profile(payload: dict) -> dict:
    """Copy ``payload`` with ``analysis_profile`` set from the request or the env."""
    raw = payload.get("analysis_profile")
    if raw is None or str(raw).strip() == "":
        chosen = configured_profile()
    else:
        chosen = normalize_profile(str(raw))
    return {**payload, "analysis_profile": chosen}


def scheduled_analysis_jobs(payload: dict) -> list[str]:
    """Return 4D jobs for this payload. ``off`` returns an empty list."""
    profile = normalize_profile(str(payload.get("analysis_profile") or ""))
    if payload.get("analysis_profile") in (None, ""):
        profile = configured_profile()
    if profile == FAST:
        return [FAST_JOB]
    if profile == DEEP:
        return [FAST_JOB, DEEP_JOB]
    return []


def queue_capacity() -> int:
    """Maximum frames held before redundant frames are coalesced."""
    return max(1, _env_int("ANALYSIS4D_QUEUE_CAPACITY", 32))


def gpu_slots() -> int:
    """How many optional GPU stages may run at once. Tracks do not take a slot."""
    return max(0, _env_int("ANALYSIS4D_GPU_SLOTS", 1))


def chunk_seconds() -> float:
    """Media seconds per causal chunk. Matches the keyframe chunk unless overridden."""
    raw = os.environ.get("ANALYSIS4D_PROFILE_CHUNK_SEC", "").strip()
    if not raw:
        raw = os.environ.get("ANALYSIS4D_CHUNK_SEC", "4")
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 4.0


def fast_event_types() -> frozenset[str]:
    """Event types whose publish lag is bounded on the fast profile."""
    raw = os.environ.get("ANALYSIS4D_FAST_EVENT_TYPES", _FAST_TYPES)
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def zone_id() -> str:
    """Site zone stamped on published envelopes."""
    return os.environ.get("ANALYSIS4D_ZONE_ID", "site").strip() or "site"


def sensor_id(mission_id: str) -> str:
    """Sensor id stamped on published envelopes. The default is the mission id."""
    raw = os.environ.get("ANALYSIS4D_SENSOR_ID", "").strip()
    return raw or mission_id


def default_prompts():
    """Prompts for a worker run. ``ANALYSIS4D_PROMPTS`` is a comma-separated list."""
    from selfsuvis.pipeline.analysis4d.providers import Prompt

    raw = os.environ.get("ANALYSIS4D_PROMPTS", "object")
    prompts = []
    for index, part in enumerate(piece.strip() for piece in raw.split(",")):
        if not part:
            continue
        prompts.append(Prompt(prompt_id=f"prompt-{index}", text=part, normalized=part.lower()))
    if prompts:
        return prompts
    return [Prompt(prompt_id="prompt-0", text="object", normalized="object")]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
