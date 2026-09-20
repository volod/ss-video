"""Mission-bundle manifests (ss-common contract `mission-bundle`).

A mission bundle is a directory holding a mission's videos, their sensor sidecars, and the
manifest `mission.json` at its root. The manifest lists every file with its digest and records
what readers otherwise re-derive: the sidecar time base, the GPS origin of the mission ENU frame,
and the recording platform.

Sidecars are found the way the pipeline reads them: `<video stem>.<kind>.jsonl` next to the video
(`pipeline.core.sidecars`), or under `sensors/` at the bundle root, plus a DJI `<video stem>.srt`
GPS file (`pipeline.media.gps`). Row times use the `t` or `timestamp` key in seconds from the
start of the video, so the time base is `media`. The origin is the first GPS fix of the first
video, which is what platform fusion and the missions `gps_origin_json` use.
"""

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from selfsuvis.pipeline.core.logging import get_logger
from selfsuvis.pipeline.core.manifests import (
    file_digest,
    relative_manifest_path,
    utc_timestamp,
    write_manifest,
)
from selfsuvis.pipeline.core.sidecars import load_jsonl_sidecar
from selfsuvis.pipeline.media.gps import _extract_from_ffprobe_atoms, _parse_srt_file
from selfsuvis.pipeline.media.subprocess_common import run_captured

logger = get_logger(__name__)

MANIFEST_NAME = "mission.json"
SENSORS_DIR = "sensors"
TIME_BASE = "media"
DEFAULT_ROBOT_ID = "robot_0"

_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MODEL_TAGS = ("com.apple.quicktime.model", "model")
_FFPROBE_TIMEOUT_SEC = 30


def probe_video(video_path: str | Path) -> dict[str, Any]:
    """Container duration, frame rate, size, creation time, and model tag from ffprobe.

    Returns only the keys ffprobe reports; an unreadable file gives an empty dict.
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
        str(video_path),
    ]  # fmt: skip
    try:
        result = run_captured(cmd, timeout=_FFPROBE_TIMEOUT_SEC, text=True)
        probe = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, ValueError) as exc:
        logger.warning("ffprobe failed for %s: %s", video_path, exc)
        return {}
    fmt = probe.get("format") or {}
    stream = next((s for s in probe.get("streams") or [] if s.get("codec_type") == "video"), {})
    info: dict[str, Any] = {}
    if _positive(fmt.get("duration")) is not None:
        info["duration_sec"] = float(fmt["duration"])
    fps = _frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    if fps is not None:
        info["fps"] = fps
    for key in ("width", "height"):
        if isinstance(stream.get(key), int) and stream[key] > 0:
            info[key] = stream[key]
    tags = {str(k).lower(): v for k, v in (fmt.get("tags") or {}).items()}
    created = _parse_time(tags.get("creation_time"))
    if created is not None:
        info["creation_time"] = created
    model = next((str(tags[t]).strip() for t in _MODEL_TAGS if str(tags.get(t, "")).strip()), "")
    if model:
        info["camera_model"] = model
    return info


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _frame_rate(value: Any) -> float | None:
    num, _, den = str(value or "").partition("/")
    rate = _positive(num)
    divisor = _positive(den or "1")
    return rate / divisor if rate is not None and divisor is not None else None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _row_times(rows: list[dict[str, Any]]) -> list[float]:
    times: list[float] = []
    for row in rows:
        value = row.get("t", row.get("timestamp"))
        if isinstance(value, int | float) and not isinstance(value, bool):
            times.append(float(value))
    return times


def _sidecar_entry(
    path: Path, root: Path, video_id: str, kind: str, fmt: str, rows: int, times: list[float]
) -> dict[str, Any]:
    sha256, size = file_digest(path)
    entry: dict[str, Any] = {
        "video_id": video_id,
        "kind": kind,
        "format": fmt,
        "path": relative_manifest_path(path, root),
        "sha256": sha256,
        "size_bytes": size,
        "row_count": rows,
    }
    if times:
        entry["t_start_sec"] = min(times)
        entry["t_end_sec"] = max(times)
    return entry


def _jsonl_sidecars(video: Path, root: Path) -> list[Path]:
    prefix = f"{video.stem}."
    folders = [video.parent, root / SENSORS_DIR]
    found: dict[Path, None] = {}
    for folder in folders:
        if folder.is_dir():
            for path in sorted(folder.glob(f"{_glob_escape(prefix)}*.jsonl")):
                found.setdefault(path.resolve(), None)
    return list(found)


def _glob_escape(text: str) -> str:
    """Escape glob metacharacters in a literal file-name prefix."""
    return re.sub(r"([*?\[])", r"[\1]", text)


def discover_sidecars(video_path: str | Path, root: str | Path) -> list[dict[str, Any]]:
    """Sidecar entries of one video: jsonl files by kind, then the DJI srt GPS file."""
    video, bundle = Path(video_path).resolve(), Path(root).resolve()
    entries: list[dict[str, Any]] = []
    for path in _jsonl_sidecars(video, bundle):
        kind = path.name[len(video.stem) + 1 : -len(".jsonl")]
        if not _KIND_RE.match(kind):
            logger.warning("Skipping sidecar with an unusable kind: %s", path)
            continue
        rows = load_jsonl_sidecar(path)
        entries.append(
            _sidecar_entry(path, bundle, video.stem, kind, "jsonl", len(rows), _row_times(rows))
        )
    srt = video.with_suffix(".srt")
    if srt.is_file():
        fixes = _parse_srt_file(str(srt))
        times = [fix["timestamp_ms"] / 1000.0 for fix in fixes]
        entries.append(_sidecar_entry(srt, bundle, video.stem, "gps", "srt", len(fixes), times))
    return entries


def first_gps_fix(video_path: str | Path) -> tuple[dict[str, float] | None, str | None]:
    """The first GPS fix of a video and its source (`srt` or `atom`), in `extract_gps` order."""
    video = Path(video_path)
    srt = video.with_suffix(".srt")
    fixes = _parse_srt_file(str(srt)) if srt.is_file() else []
    if fixes:
        fix, source = fixes[0], "srt"
    else:
        fix, source = _extract_from_ffprobe_atoms(str(video)), "atom"
    if not fix:
        return None, None
    origin = {"lat": float(fix["lat"]), "lon": float(fix["lon"]), "alt": float(fix.get("alt", 0.0))}
    return origin, source


def _video_entry(
    video: Path, root: Path, probe: dict[str, Any], gps_source: str | None
) -> dict[str, Any]:
    sha256, size = file_digest(video)
    entry: dict[str, Any] = {
        "video_id": video.stem,
        "path": relative_manifest_path(video, root),
        "sha256": sha256,
        "size_bytes": size,
    }
    for key in ("duration_sec", "fps", "width", "height"):
        if key in probe:
            entry[key] = probe[key]
    if gps_source:
        entry["gps_source"] = gps_source
    return entry


def build_mission_bundle(
    videos: Sequence[str | Path],
    *,
    root: str | Path,
    mission_id: str | None = None,
    robot_id: str = DEFAULT_ROBOT_ID,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """A `mission-bundle` manifest for videos under `root` and their sidecars.

    `mission_id` defaults to the first video's stem, as a local run names its output directory.
    """
    if not videos:
        raise ValueError("a mission bundle needs at least one video")
    bundle = Path(root).resolve()
    paths = [Path(video).resolve() for video in videos]
    entries, sidecars, origin = [], [], None
    first_probe: dict[str, Any] = {}
    for index, video in enumerate(paths):
        probe = probe_video(video)
        fix, gps_source = first_gps_fix(video)
        if index == 0:
            first_probe, origin = probe, fix
        entries.append(_video_entry(video, bundle, probe, gps_source))
        sidecars.extend(discover_sidecars(video, bundle))
    manifest: dict[str, Any] = {
        "mission_id": mission_id or paths[0].stem,
        "created_at": utc_timestamp(created_at or datetime.now(UTC)),
        "time_base": TIME_BASE,
    }
    if "creation_time" in first_probe:
        manifest["start_time"] = utc_timestamp(first_probe["creation_time"])
    platform = {"robot_id": robot_id}
    if "camera_model" in first_probe:
        platform["camera_model"] = first_probe["camera_model"]
    manifest["platform"] = platform
    if origin is not None:
        manifest["origin"] = origin
    manifest["videos"] = entries
    manifest["sidecars"] = sidecars
    return manifest


def write_mission_bundle(manifest: dict[str, Any], root: str | Path) -> Path:
    """Write the manifest as `mission.json` at the bundle root."""
    return write_manifest(manifest, Path(root) / MANIFEST_NAME)
