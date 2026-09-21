"""Keyframe and track benchmark.

``python -m selfsuvis.pipeline.analysis4d.track_benchmark`` checks the fixture
corpus, the occlusion/cut identity rule, and, on a CUDA host, the provider
gates. ``passed`` requires keyframe coverage, monotonic tracks, a stable seed,
and no silent id reuse. The GPU probe is part of the declared benchmark; unit
tests call the same function with the probe disabled.
"""

import argparse
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from selfsuvis.pipeline.analysis4d.benchmark import default_corpus
from selfsuvis.pipeline.analysis4d.io import write_bytes
from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal, SelectorConfig, select_keyframes
from selfsuvis.pipeline.analysis4d.passes import write_profiles
from selfsuvis.pipeline.analysis4d.pin import PinChoice, ProviderProbe, choose_pin
from selfsuvis.pipeline.analysis4d.providers import Prompt, ScriptedGrounding, probe_providers
from selfsuvis.pipeline.analysis4d.schemas import MissionTruth
from selfsuvis.pipeline.analysis4d.tracks import Detection
from selfsuvis.pipeline.analysis4d.validate import validate_bundle
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

SCHEMA_TRACK_BENCHMARK = "ss-video.track-benchmark.v1"
_COVERAGE_SEC = 0.5


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrackBenchmarkReport(_Model):
    """Machine-readable result of the keyframe and track benchmark."""

    schema_version: str = SCHEMA_TRACK_BENCHMARK
    passed: bool
    keyframe_event_coverage: float | None = None
    monotonic: bool
    seed_stable: bool
    silent_id_reuse: bool
    profiles: list[str] = Field(default_factory=lambda: ["fast", "deep"])
    pin: PinChoice | None = None
    probes: list[ProviderProbe] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


def signals_for_truth(truth: MissionTruth, *, step: float = 1.0) -> list[FrameSignal]:
    """Build a drift spike at every annotated keyframe boundary."""
    grid: list[float] = []
    stamp = 0.0
    while stamp <= truth.media_duration_sec + 1e-9:
        grid.append(round(stamp, 3))
        stamp += step
    boundaries = [round(value, 3) for value in truth.keyframe_boundaries_sec]
    times = sorted(set(grid) | set(boundaries))
    frames: list[FrameSignal] = []
    previous = (1.0, 0.0, 0.0, 0.0)
    for value in times:
        embedding = previous
        drift = 0.0
        hit = any(abs(value - boundary) <= 1e-6 for boundary in boundaries) and value > 0
        if hit:
            embedding = (0.0, 1.0, 0.0, 0.0) if previous[0] > 0.5 else (1.0, 0.0, 0.0, 0.0)
            drift = 1.0
        frames.append(FrameSignal(t_sec=value, embedding=embedding, drift=drift))
        previous = embedding
    return frames


def run_track_benchmark(
    corpus: Path | None = None,
    output: Path | None = None,
    *,
    probe_gpu: bool = False,
    seed: int = 0,
) -> TrackBenchmarkReport:
    """Score keyframe coverage, identity, and the optional provider probe.

    Args:
        corpus: Pinned corpus directory. Defaults to ``tests/assets/analysis4d``.
        output: Directory for the occlusion bundle. The report itself is written
            by :func:`main`.
        probe_gpu: When true, load candidate providers and apply the pin gate.
        seed: Selector and tracker seed.

    Returns:
        The report. ``passed`` is false when any fixture or, if probed, model gate fails.
    """
    root = Path(corpus) if corpus is not None else default_corpus()
    failures: list[str] = []
    covered = 0
    total = 0
    config = SelectorConfig(seed=seed)
    for mission_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        truth_path = mission_dir / "truth.json"
        if (
            not truth_path.is_file()
            or mission_dir.parent.name == "invalid"
            or "invalid" in mission_dir.parts
        ):
            continue
        if mission_dir.name == "invalid":
            continue
        truth = MissionTruth.model_validate_json(truth_path.read_text(encoding="utf-8"))
        if not truth.keyframe_boundaries_sec:
            continue
        signals = signals_for_truth(truth)
        for profile in ("fast", "deep"):
            picks, _skips = select_keyframes(signals, config, profile=profile)
            times = [pick.t_sec for pick in picks]
            for boundary in truth.keyframe_boundaries_sec:
                total += 1
                if any(abs(stamp - boundary) <= _COVERAGE_SEC for stamp in times):
                    covered += 1
                else:
                    failures.append(f"{truth.mission_id} {profile} missed boundary {boundary}")
    coverage = None if total == 0 else covered / total
    if coverage is not None and coverage < 1:
        failures.append(f"keyframe_event_coverage {coverage}")

    monotonic, seed_stable, silent = True, True, False
    if output is not None:
        bundle = _occlusion_bundle(Path(output) / "occlusion-cut", seed)
        validate_bundle(bundle)
        silent = _silent_reuse(bundle)
        monotonic = _monotonic(bundle)
        seed_stable = _seed_stable(seed)
        if silent:
            failures.append("occlusion/cut reused a track id without an identity link")
        if not _deep_link(bundle):
            failures.append("deep pass did not link the cut to the earlier track")
        if not monotonic:
            failures.append("track timestamps are not monotonic")
        if not seed_stable:
            failures.append("track output changed under a fixed seed")
    else:
        silent = _silent_reuse_memory(seed)
        seed_stable = _seed_stable(seed)
        if silent:
            failures.append("occlusion/cut reused a track id without an identity link")
        if not seed_stable:
            failures.append("track output changed under a fixed seed")

    probes: list[ProviderProbe] = []
    pin: PinChoice | None = None
    if probe_gpu:
        logger.info("Probing 4D providers")
        probes = probe_providers()
        pin = choose_pin(probes)
        from selfsuvis.pipeline.analysis4d.pin import (
            PINNED_GROUNDING,
            PINNED_MASK,
            PINNED_REVISION,
            PINNED_WEIGHTS_DIGEST,
        )
        from selfsuvis.pipeline.analysis4d.providers import GROUNDING_DINO_ID, _weights_digest

        if pin.model_gate != "pass":
            failures.append(pin.detail)
        elif pin.grounding != PINNED_GROUNDING or pin.mask != PINNED_MASK or pin.count != "":
            failures.append(
                f"live pin {pin.detail} does not match {PINNED_GROUNDING}+{PINNED_MASK}"
            )
        else:
            revision, digest = _weights_digest(GROUNDING_DINO_ID)
            if revision != PINNED_REVISION or digest != PINNED_WEIGHTS_DIGEST:
                failures.append(f"cached weights {revision} {digest} do not match the pin")
    return TrackBenchmarkReport(
        passed=not failures,
        keyframe_event_coverage=coverage,
        monotonic=monotonic,
        seed_stable=seed_stable,
        silent_id_reuse=silent,
        pin=pin,
        probes=probes,
        failures=failures,
    )


def _occlusion_frames() -> tuple[list[FrameSignal], ScriptedGrounding]:
    frames = [
        FrameSignal(t_sec=0.0, embedding=(1.0, 0.0), drift=1.0),
        FrameSignal(t_sec=5.0, embedding=(1.0, 0.0)),
        FrameSignal(t_sec=9.0, embedding=(0.0, 1.0), drift=1.0, scene_cut=True),
    ]
    appearance = (1.0, 0.0)
    left = Detection(
        prompt_id="prompt-crate",
        label_raw="crate",
        label_normalized="crate",
        xywh=(0.10, 0.10, 0.20, 0.20),
        confidence=0.9,
        appearance=appearance,
    )
    right = Detection(
        prompt_id="prompt-crate",
        label_raw="crate",
        label_normalized="crate",
        xywh=(0.60, 0.60, 0.20, 0.20),
        confidence=0.9,
        appearance=appearance,
    )
    return frames, ScriptedGrounding({0.0: [left], 9.0: [right]})


def _occlusion_bundle(dest: Path, seed: int) -> Path:
    frames, grounding = _occlusion_frames()
    prompts = [Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")]
    return write_profiles(
        "mission-occlusion-run",
        frames,
        prompts,
        dest,
        grounding=grounding,
        selector=SelectorConfig(seed=seed, max_gap_sec=30.0),
        created_at="2026-01-15T00:00:00Z",
    )


def _silent_reuse(bundle: Path) -> bool:
    validated = validate_bundle(bundle)
    deep = [row for row in validated.tracks if row.observation_id.startswith("obs-deep-")]
    return _reuses(deep)


def _silent_reuse_memory(seed: int) -> bool:
    from selfsuvis.pipeline.analysis4d.passes import run_pass

    frames, grounding = _occlusion_frames()
    prompts = [Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")]
    output = run_pass(
        "mission-occlusion-run",
        frames,
        prompts,
        profile="deep",
        grounding=grounding,
        selector=SelectorConfig(seed=seed, max_gap_sec=30.0),
    )
    start = next(
        row for row in output.observations if row.t_sec == 0.0 and row.state == "tentative"
    )
    later = [row for row in output.observations if row.t_sec == 9.0 and row.state == "tentative"]
    if not later:
        return True
    return later[-1].track_id == start.track_id


def _reuses(rows: list) -> bool:
    start = [row for row in rows if row.t_sec == 0.0 and row.state == "tentative"]
    later = [row for row in rows if row.t_sec == 9.0 and row.state == "tentative"]
    if not start or not later:
        return True
    return later[-1].track_id == start[0].track_id


def _deep_link(bundle: Path) -> bool:
    validated = validate_bundle(bundle)
    deep = [row for row in validated.tracks if row.observation_id.startswith("obs-deep-")]
    start = [row for row in deep if row.t_sec == 0.0 and row.state == "tentative"]
    later = [row for row in deep if row.t_sec == 9.0 and row.state == "tentative"]
    if not start or not later:
        return False
    link = later[-1].identity_link
    fast = [row for row in validated.tracks if row.observation_id.startswith("obs-fast-")]
    fast_later = [row for row in fast if row.t_sec == 9.0 and row.state == "tentative"]
    if any(row.identity_link is not None for row in fast_later):
        return False
    return (
        link is not None
        and link.linked_track_id == start[0].track_id
        and link.reason == "camera_cut"
    )


def _monotonic(bundle: Path) -> bool:
    validated = validate_bundle(bundle)
    times = [row.t_sec for row in validated.tracks]
    return times == sorted(times)


def _seed_stable(seed: int) -> bool:
    from selfsuvis.pipeline.analysis4d.passes import run_pass

    frames, grounding = _occlusion_frames()
    prompts = [Prompt(prompt_id="prompt-crate", text="crate", normalized="crate")]

    def once(profile: str) -> list[tuple]:
        output = run_pass(
            "mission-occlusion-run",
            frames,
            prompts,
            profile=profile,
            grounding=grounding,
            selector=SelectorConfig(seed=seed, max_gap_sec=30.0),
        )
        return [
            (row.track_id, row.state, row.t_sec, row.identity_link_id)
            for row in output.observations
        ]

    return once("fast") == once("fast") and once("deep") == once("deep")


def default_track_report_path() -> Path:
    """Return ``$DATA_DIR/analysis/_benchmark/tracks-report.json``."""
    from selfsuvis.pipeline.analysis4d.paths import benchmark_report_path

    return benchmark_report_path().with_name("tracks-report.json")


def main(argv: list[str] | None = None) -> int:
    """Write the track benchmark report. Returns 0 when it passes."""
    parser = argparse.ArgumentParser(description="4D keyframe and track benchmark")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report_path = args.output or default_track_report_path()
    bundle_dir = args.bundle_dir or report_path.parent / "tracks"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    report = run_track_benchmark(
        args.corpus,
        bundle_dir,
        probe_gpu=not args.skip_gpu,
        seed=args.seed,
    )
    payload = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    write_bytes(report_path, payload.encode("utf-8"))
    logger.info("track benchmark passed=%s report=%s", report.passed, report_path)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
