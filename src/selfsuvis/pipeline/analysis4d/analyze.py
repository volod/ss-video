"""Analyze one video file into a 4D mission directory.

``python -m selfsuvis.pipeline.analysis4d.analyze VIDEO`` samples the file,
runs the fast profile with the pinned grounding model, and prints where the
tracks and timeline were written.
"""

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from selfsuvis.pipeline.analysis4d.io import read_jsonl, read_model, write_bytes
from selfsuvis.pipeline.analysis4d.profile import default_prompts, normalize_profile
from selfsuvis.pipeline.analysis4d.providers import Prompt
from selfsuvis.pipeline.analysis4d.schemas import AnalysisManifest, SceneTimeline, TrackRecord
from selfsuvis.pipeline.analysis4d.signals import signals_from_paths
from selfsuvis.pipeline.core import get_logger, settings
from selfsuvis.pipeline.workflows.analysis4d_profile import run_deep_profile, run_fast_profile

logger = get_logger(__name__)

SUMMARY_SCHEMA = "ss-video.analyze-summary.v1"
_DEFAULT_FPS = 1.0


def mission_id_for(video: Path) -> str:
    """Return a mission id derived from the video file name."""
    cleaned = "".join(char if char.isalnum() else "-" for char in video.stem.lower())
    slug = "-".join(part for part in cleaned.split("-") if part)
    return f"mission-{slug or 'video'}"[:128]


def extract_sampled_frames(video: Path, dest: Path, fps: float) -> list[tuple[float, str]]:
    """Write JPEGs at ``fps`` and return ``(t_sec, path)`` in decode order.

    Args:
        video: Input video path.
        dest: Directory that receives ``frame_######.jpg``. Existing frames in
            that directory are removed first.
        fps: Output frame rate. Values below 0.1 are raised to 0.1.

    Returns:
        Timestamped JPEG paths. Empty when ffmpeg writes no frames.
    """
    from selfsuvis.pipeline.media.subprocess_common import run_checked

    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("frame_*.jpg"):
        old.unlink()
    rate = max(0.1, fps)
    pattern = str(dest / "frame_%06d.jpg")
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={rate},format=yuv420p",
            "-color_range",
            "2",
            "-q:v",
            "3",
            pattern,
        ],
        timeout=int(settings.FFMPEG_TIMEOUT_SEC),
    )
    files = sorted(dest.glob("frame_*.jpg"))
    return [(index / rate, str(path)) for index, path in enumerate(files)]


def analyze_video(
    video: Path,
    *,
    mission_id: str | None = None,
    profile: str = "fast",
    prompts: list[Prompt] | None = None,
    fps: float = _DEFAULT_FPS,
    dest: Path | None = None,
    decoder: Callable[[Path, Path, float], list[tuple[float, str]]] | None = None,
    grounding: object | None = None,
    fixture_geometry: bool = False,
    slots: int | None = None,
) -> dict:
    """Sample ``video`` and run one 4D profile.

    Args:
        video: Existing video file.
        mission_id: Artifact name. The default comes from the file name.
        profile: ``fast`` or ``deep``. Any other value runs ``fast``.
        prompts: Grounding prompts. The default is ``ANALYSIS4D_PROMPTS``.
        fps: Sample rate passed to the decoder.
        dest: 4D artifact directory. The default is the mission analysis dir.
        decoder: Replacement for ffmpeg. It receives ``(video, frame_dir, fps)``.
        grounding: Detector. The default is the pinned grounding provider.
        fixture_geometry: Write synthetic boxes. Leave this false for a real file.
        slots: GPU slots for optional stages. The default is ``ANALYSIS4D_GPU_SLOTS``.

    Returns:
        A summary of frames, tracks, accepted events, and artifact paths.

    Raises:
        FileNotFoundError: ``video`` does not exist.
        ValueError: The decoder returned no frames.
    """
    source = Path(video)
    if not source.is_file():
        raise FileNotFoundError(source)
    chosen = normalize_profile(profile)
    if chosen == "off":
        chosen = "fast"
    mission = mission_id or mission_id_for(source)
    root = settings.data_dir() / "analysis" / mission
    target = Path(dest) if dest is not None else root / "4d"
    frame_dir = target.parent / "frames"
    decode = decoder or extract_sampled_frames
    samples = list(decode(source, frame_dir, fps))
    if not samples:
        raise ValueError(f"no frames decoded from {source}")
    frames = signals_from_paths(samples)
    active_prompts = list(prompts) if prompts is not None else default_prompts()
    _install_regions(source, target.parent)
    kwargs = {
        "dest": target,
        "grounding": grounding,
        "fixture_geometry": fixture_geometry,
        "duration_sec": max(frames[-1].t_sec, 1.0 / max(fps, 0.1)),
        "slots": slots,
    }
    if chosen == "deep":
        run_deep_profile(mission, frames, active_prompts, **kwargs)
    else:
        run_fast_profile(mission, frames, active_prompts, **kwargs)
    summary = _summary(mission, chosen, target, frame_count=len(frames))
    summary_path = target.parent / "summary.json"
    write_bytes(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    summary["summary_path"] = str(summary_path)
    logger.info(
        "analyze mission=%s profile=%s tracks=%d events=%d",
        mission,
        chosen,
        summary["track_count"],
        summary["accepted_event_count"],
    )
    return summary


def _summary(mission_id: str, profile: str, dest: Path, *, frame_count: int) -> dict:
    tracks = (
        read_jsonl(dest / "tracks.jsonl", TrackRecord) if (dest / "tracks.jsonl").is_file() else []
    )
    timeline = (
        read_model(dest / "timeline.json", SceneTimeline)
        if (dest / "timeline.json").is_file()
        else None
    )
    labels = Counter(row.label_normalized for row in tracks)
    accepted = []
    if timeline is not None:
        accepted = [
            {
                "event_id": event.event_id,
                "type": event.type,
                "summary": event.summary,
                "start_sec": event.start_sec,
                "end_sec": event.end_sec,
                "participants": list(event.participants),
            }
            for event in timeline.events
            if event.verification.status == "accepted"
        ]
    geometry = dest / "geometry"
    geometry_count = len(list(geometry.rglob("*.json"))) if geometry.is_dir() else 0
    manifest = (
        read_model(dest / "manifest.json", AnalysisManifest)
        if (dest / "manifest.json").is_file()
        else None
    )
    return {
        "schema_version": SUMMARY_SCHEMA,
        "mission_id": mission_id,
        "profile": profile,
        "frame_count": frame_count,
        "track_count": len({row.track_id for row in tracks}),
        "observation_count": len(tracks),
        "labels": dict(sorted(labels.items())),
        "accepted_event_count": len(accepted),
        "accepted_events": accepted,
        "geometry_samples": geometry_count,
        "metric_scale": (
            manifest.coordinate_frame.metric_scale if manifest is not None else "unavailable"
        ),
        "degradations": (
            [f"{item.stage}:{item.code}" for item in manifest.degradations]
            if manifest is not None
            else []
        ),
        "artifact_dir": str(dest),
        "timeline_path": str(dest / "timeline.json"),
        "tracks_path": str(dest / "tracks.jsonl"),
    }


def _print_summary(summary: dict) -> None:
    print(f"mission_id={summary['mission_id']}")
    print(f"profile={summary['profile']}")
    print(f"frames={summary['frame_count']}")
    print(f"tracks={summary['track_count']}")
    print(f"labels={json.dumps(summary['labels'], sort_keys=True)}")
    print(f"accepted_events={summary['accepted_event_count']}")
    print(f"geometry_samples={summary['geometry_samples']}")
    print(f"metric_scale={summary['metric_scale']}")
    print(f"degradations={json.dumps(summary['degradations'])}")
    for event in summary["accepted_events"][:20]:
        print(
            f"event {event['event_id']} type={event['type']} "
            f"t={event['start_sec']:.3f}-{event['end_sec']:.3f} {event['summary']}"
        )
    print(f"artifacts={summary['artifact_dir']}")
    print(f"summary={summary['summary_path']}")


def _install_regions(video: Path, mission_root: Path) -> None:
    """Copy ``<video-stem>.regions.json`` next to the mission artifacts."""
    source = video.with_name(video.stem + ".regions.json")
    target = mission_root / "regions.json"
    if not source.is_file() or target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one video into a 4D mission directory")
    parser.add_argument("video", type=Path, help="Path to a video file")
    parser.add_argument("--profile", default="fast", choices=("fast", "deep"))
    parser.add_argument(
        "--prompts",
        default="person,vehicle",
        help="Comma-separated grounding prompts",
    )
    parser.add_argument("--fps", type=float, default=_DEFAULT_FPS)
    parser.add_argument("--mission-id", default=None)
    args = parser.parse_args()
    prompts = []
    for index, part in enumerate(piece.strip() for piece in args.prompts.split(",")):
        if not part:
            continue
        prompts.append(Prompt(prompt_id=f"prompt-{index}", text=part, normalized=part.lower()))
    if not prompts:
        prompts = default_prompts()
    try:
        summary = analyze_video(
            args.video,
            mission_id=args.mission_id,
            profile=args.profile,
            prompts=prompts,
            fps=args.fps,
        )
    except FileNotFoundError as exc:
        if exc.filename in (None, "ffmpeg") or "ffmpeg" in str(exc):
            raise SystemExit("ffmpeg is required to analyze a video") from exc
        raise SystemExit(f"video not found: {args.video}") from exc
    _print_summary(summary)


if __name__ == "__main__":
    main()
