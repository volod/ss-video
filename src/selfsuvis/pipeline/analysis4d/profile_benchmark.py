"""15-minute fast-profile load check.

``python -m selfsuvis.pipeline.analysis4d.profile_benchmark`` runs a 900-second
scripted stream through the production scheduler on this host. It records the
real-time factor, queue depth, verified-event lag, gap coverage, restart
replay, and fast-to-deep supersession.
"""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from selfsuvis.pipeline.analysis4d.io import read_jsonl, read_model, write_bytes
from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal
from selfsuvis.pipeline.analysis4d.providers import Prompt, ScriptedGrounding
from selfsuvis.pipeline.analysis4d.schemas import AnalysisManifest, GapRecord, SceneTimeline
from selfsuvis.pipeline.analysis4d.tracks import Detection
from selfsuvis.pipeline.core import get_logger
from selfsuvis.pipeline.workflows.analysis4d_profile import run_deep_profile, run_fast_profile

logger = get_logger(__name__)

DURATION_SEC = 900.0
FPS = 2.0
CAPACITY = 4
SCHEMA = "ss-video.profile-benchmark.v1"


def reference_device() -> str:
    """Return the CUDA device name, or ``cpu`` when CUDA is not available."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return str(torch.cuda.get_device_name(0))


def build_stream(duration_sec: float = DURATION_SEC, fps: float = FPS):
    """Return frames, a scripted detector, and one prompt for ``duration_sec``."""
    step = 1.0 / fps
    count = int(duration_sec * fps)
    frames: list[FrameSignal] = []
    hits: dict[float, list[Detection]] = {}
    detection = Detection(
        prompt_id="prompt-crate",
        label_raw="crate",
        label_normalized="crate",
        xywh=(0.2, 0.2, 0.2, 0.2),
        confidence=0.9,
        appearance=(1.0, 0.0),
    )
    for index in range(count):
        stamp = round(index * step, 6)
        forced = index % int(fps * 30) == 0
        frames.append(
            FrameSignal(
                t_sec=stamp,
                embedding=(1.0, 0.0),
                scene_cut=forced,
                quality_ok=True,
            )
        )
        if forced:
            hits[stamp] = [detection]
    prompt = Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")
    return frames, ScriptedGrounding(hits), [prompt]


def run_profile_benchmark(
    dest: Path,
    *,
    duration_sec: float = DURATION_SEC,
    report_path: Path | None = None,
) -> dict:
    """Run the load fixture and write a JSON report.

    Args:
        dest: Directory for the mission artifacts.
        duration_sec: Media length. The acceptance run uses 900 seconds.
        report_path: Where to write the report. The default is next to ``dest``.

    Returns:
        The report object. ``passed`` is false when a gate misses its limit.
    """
    frames, grounding, prompts = build_stream(duration_sec)
    stamp = datetime.now(UTC).replace(microsecond=0)
    common = {
        "grounding": grounding,
        "fixture_geometry": True,
        "capacity": CAPACITY,
        "slots": 0,
        "published_at": stamp,
        "duration_sec": duration_sec,
    }
    fast = run_fast_profile("mission-load", frames, prompts, dest=dest, **common)
    replay = run_fast_profile("mission-load", frames, prompts, dest=dest, **common)
    deep = run_deep_profile("mission-load", frames, prompts, dest=dest, **common)
    deep_replay = run_deep_profile("mission-load", frames, prompts, dest=dest, **common)
    gaps = read_jsonl(dest / "gaps.jsonl", GapRecord)
    queue_gaps = [gap for gap in gaps if gap.reason == "queue_coalesce"]
    timeline = read_model(dest / "timeline.json", SceneTimeline)
    manifest = read_model(dest / "manifest.json", AnalysisManifest)
    fast_ids = set(fast.published_event_ids)
    p95 = _percentile(fast.lags_sec, 95)
    failures: list[str] = []
    if fast.real_time_factor > 1.0:
        failures.append(f"real_time_factor {fast.real_time_factor:.4f} exceeds 1.0")
    if fast.queue_depth > CAPACITY:
        failures.append(f"queue_depth {fast.queue_depth} exceeds {CAPACITY}")
    if not fast.lags_sec or p95 > 3.0:
        failures.append(f"p95 lag {p95} exceeds 3 seconds or no fast events were published")
    if not queue_gaps:
        failures.append("discarded frames produced no queue gap")
    if any(gap.end_sec <= gap.start_sec for gap in queue_gaps):
        failures.append("a queue gap has an empty interval")
    if not replay.replayed or not deep_replay.replayed:
        failures.append("restart replay published or rebuilt a finished profile")
    if not deep.superseded or manifest.profile != "deep" or not manifest.supersedes_sha256:
        failures.append("deep profile did not supersede the fast manifest")
    if not any(event.event_id in fast_ids for event in timeline.events):
        failures.append("deep timeline dropped the fast events")
    if timeline.profile != "deep":
        failures.append("timeline profile is not deep")
    report = {
        "schema_version": SCHEMA,
        "passed": not failures,
        "reference_device": reference_device(),
        "duration_sec": duration_sec,
        "frame_count": len(frames),
        "real_time_factor": fast.real_time_factor,
        "queue_depth": fast.queue_depth,
        "queue_capacity": CAPACITY,
        "p95_lag_sec": p95,
        "fast_event_count": len(fast.lags_sec),
        "queue_gap_count": len(queue_gaps),
        "replayed": replay.replayed and deep_replay.replayed,
        "supersedes_sha256": manifest.supersedes_sha256,
        "degradations": fast.degradations,
        "backlog": fast.backlog,
        "failures": failures,
    }
    target = report_path or dest.parent / "profile-report.json"
    write_bytes(target, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    logger.info(
        "profile benchmark passed=%s rtf=%.4f p95_lag=%.4f queue=%d device=%s",
        report["passed"],
        report["real_time_factor"],
        report["p95_lag_sec"],
        report["queue_depth"],
        report["reference_device"],
    )
    return report


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percent / 100 * len(ordered)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 15-minute 4D profile load check")
    parser.add_argument("--duration-sec", type=float, default=DURATION_SEC)
    parser.add_argument("--dest", type=Path, default=None)
    args = parser.parse_args()
    from selfsuvis.pipeline.core import settings

    root = args.dest or (settings.data_dir() / "analysis" / "_benchmark" / "profile-load")
    report = run_profile_benchmark(
        root,
        duration_sec=args.duration_sec,
        report_path=settings.data_dir() / "analysis" / "_benchmark" / "profile-report.json",
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
