"""Bounded MaxInfo-style keyframe selector.

The live path keeps a pairwise histogram/SSIM/drift gate, then fills a fixed
per-chunk budget with a greedy maximum-volume pick. Each added frame is the
one with the largest residual against the span of frames already chosen.
The deep profile reselects over the whole sequence with a budget scaled by
the chunk count. Scene cuts, prompt changes, calibration changes, stream
start, and a maximum time gap are forced and can exceed that budget.
"""

from dataclasses import dataclass

_FORCED = frozenset(
    {
        "stream_start",
        "scene_cut",
        "prompt_change",
        "calibration_change",
        "max_gap",
        "track_birth",
        "track_death",
        "low_association",
        "count_disagreement",
    }
)


@dataclass(frozen=True)
class FrameSignal:
    """One decoded frame. Embeddings are compared only within one sequence."""

    t_sec: float
    embedding: tuple[float, ...] = ()
    hist_diff: float = 0.0
    ssim_diff: float = 0.0
    drift: float = 0.0
    scene_cut: bool = False
    prompt_changed: bool = False
    calibration_changed: bool = False
    quality_ok: bool = True
    image: object | None = None


@dataclass(frozen=True)
class SelectorConfig:
    """Thresholds for the cheap gate, the chunk budget, and the heartbeat."""

    hist_thresh: float = 0.25
    ssim_thresh: float = 0.25
    embed_drift_thresh: float = 0.15
    max_gap_sec: float = 10.0
    chunk_sec: float = 4.0
    keyframe_budget: int = 2
    min_spacing_sec: float = 0.25
    seed: int = 0


@dataclass(frozen=True)
class KeyframePick:
    """A kept timestamp. ``forced`` is true when a trigger required the frame."""

    t_sec: float
    reasons: tuple[str, ...]
    forced: bool

    def with_reason(self, reason: str) -> "KeyframePick":
        """Return a copy that includes ``reason``."""
        if reason in self.reasons:
            return self
        reasons = tuple(sorted((*self.reasons, reason)))
        return KeyframePick(
            t_sec=self.t_sec, reasons=reasons, forced=self.forced or reason in _FORCED
        )


@dataclass(frozen=True)
class SkipSpan:
    """Discarded frames. The caller turns these into gap records."""

    start_sec: float
    end_sec: float
    skipped_frames: int
    reason: str


def select_keyframes(
    frames: list[FrameSignal],
    config: SelectorConfig | None = None,
    *,
    profile: str = "fast",
) -> tuple[list[KeyframePick], list[SkipSpan]]:
    """Select keyframes in timestamp order.

    Args:
        frames: Decoded frames. The list is copied and sorted by time.
        config: Gate and budget. ``seed`` breaks exact residual ties only.
        profile: ``fast`` selects inside chunks. ``deep`` reselects globally.

    Returns:
        Picks and skip spans. Picks are non-decreasing in ``t_sec``.
    """
    cfg = config or SelectorConfig()
    ordered = sorted(frames, key=lambda frame: (frame.t_sec, id(frame)))
    skips = _quality_skips(ordered)
    usable = [frame for frame in ordered if frame.quality_ok and frame.t_sec >= 0]
    if not usable:
        return [], skips

    first = usable[0]
    candidates = [frame for frame in usable if _pairwise(frame, first, cfg)]
    reasons: dict[int, list[str]] = {}
    frames_by_id = {id(frame): frame for frame in usable}
    for frame in candidates:
        triggered = _trigger_reasons(frame, first)
        if triggered:
            reasons[id(frame)] = triggered

    if profile == "deep":
        groups = [candidates]
        budgets = [_deep_budget(cfg, usable)]
    else:
        groups = _chunks(candidates, first.t_sec, cfg.chunk_sec)
        budgets = [max(1, cfg.keyframe_budget) for _ in groups]

    for group, budget in zip(groups, budgets, strict=True):
        for frame, added in _maxinfo(group, budget, reasons, cfg.seed):
            bucket = reasons.setdefault(id(frame), [])
            for reason in added:
                if reason not in bucket:
                    bucket.append(reason)

    _heartbeat(usable, reasons, frames_by_id, cfg.max_gap_sec)
    picks = [
        KeyframePick(
            t_sec=frames_by_id[ident].t_sec,
            reasons=tuple(sorted(bucket)),
            forced=any(reason in _FORCED for reason in bucket),
        )
        for ident, bucket in reasons.items()
        if bucket
    ]
    picks.sort(key=lambda pick: pick.t_sec)
    picks, coalesced = _coalesce(picks, cfg.min_spacing_sec)
    return picks, [*skips, *coalesced]


def _deep_budget(config: SelectorConfig, usable: list[FrameSignal]) -> int:
    span = max(0.0, usable[-1].t_sec - usable[0].t_sec)
    chunks = max(1, int(span / config.chunk_sec) + 1) if config.chunk_sec > 0 else 1
    return max(1, config.keyframe_budget) * chunks


def _pairwise(frame: FrameSignal, first: FrameSignal, config: SelectorConfig) -> bool:
    if frame is first:
        return True
    if frame.scene_cut or frame.prompt_changed or frame.calibration_changed:
        return True
    return (
        frame.hist_diff > config.hist_thresh
        or frame.ssim_diff > config.ssim_thresh
        or frame.drift > config.embed_drift_thresh
    )


def _trigger_reasons(frame: FrameSignal, first: FrameSignal) -> list[str]:
    reasons: list[str] = []
    if frame is first:
        reasons.append("stream_start")
    if frame.scene_cut:
        reasons.append("scene_cut")
    if frame.prompt_changed:
        reasons.append("prompt_change")
    if frame.calibration_changed:
        reasons.append("calibration_change")
    return reasons


def _chunks(frames: list[FrameSignal], origin: float, chunk_sec: float) -> list[list[FrameSignal]]:
    if not frames:
        return []
    if chunk_sec <= 0:
        return [list(frames)]
    grouped: dict[int, list[FrameSignal]] = {}
    for frame in frames:
        index = int((frame.t_sec - origin) / chunk_sec)
        grouped.setdefault(max(0, index), []).append(frame)
    return [grouped[index] for index in sorted(grouped)]


def _maxinfo(
    frames: list[FrameSignal],
    budget: int,
    forced: dict[int, list[str]],
    seed: int,
) -> list[tuple[FrameSignal, list[str]]]:
    """Greedy maximum-volume selection inside one pool."""
    chosen: list[tuple[FrameSignal, list[str]]] = []
    basis: list[tuple[float, ...]] = []
    required = [frame for frame in frames if id(frame) in forced]
    pool = [frame for frame in frames if id(frame) not in forced]
    for frame in required:
        _take(chosen, basis, frame, [])
    while len(chosen) < budget and pool:
        ranked = sorted(pool, key=lambda frame: _rank(frame, basis, seed))
        residual = _residual(ranked[0].embedding, basis)
        if chosen and residual <= 1e-8:
            break
        frame = ranked[0]
        pool.remove(frame)
        _take(chosen, basis, frame, ["diversity"])
    return chosen


def _take(
    chosen: list[tuple[FrameSignal, list[str]]],
    basis: list[tuple[float, ...]],
    frame: FrameSignal,
    reasons: list[str],
) -> None:
    residual, unit = _unit_residual(frame.embedding, basis)
    if residual > 1e-8:
        basis.append(unit)
    chosen.append((frame, reasons))


def _rank(
    frame: FrameSignal, basis: list[tuple[float, ...]], seed: int
) -> tuple[float, float, int]:
    residual = round(_residual(frame.embedding, basis), 9)
    tie = ((seed + 1) * ((id(frame) % 1_000_003) + 1)) % 1_000_000_007
    return (-residual, frame.t_sec, tie)


def _normalize(embedding: tuple[float, ...]) -> list[float]:
    if not embedding:
        return [0.0]
    norm = sum(value * value for value in embedding) ** 0.5
    if norm <= 1e-12:
        return [0.0 for _ in embedding]
    return [value / norm for value in embedding]


def _unit_residual(
    embedding: tuple[float, ...], basis: list[tuple[float, ...]]
) -> tuple[float, tuple[float, ...]]:
    acc = _normalize(embedding)
    width = len(acc)
    for basis_vec in basis:
        padded = list(basis_vec) + [0.0] * (width - len(basis_vec))
        dot = sum(left * right for left, right in zip(acc, padded, strict=False))
        acc = [left - dot * right for left, right in zip(acc, padded, strict=False)]
    norm = sum(value * value for value in acc) ** 0.5
    if norm <= 1e-12:
        return 0.0, tuple(0.0 for _ in acc)
    return norm, tuple(value / norm for value in acc)


def _residual(embedding: tuple[float, ...], basis: list[tuple[float, ...]]) -> float:
    residual, _unit = _unit_residual(embedding, basis)
    return residual


def _heartbeat(
    usable: list[FrameSignal],
    reasons: dict[int, list[str]],
    frames_by_id: dict[int, FrameSignal],
    max_gap_sec: float,
) -> None:
    """Force a quality-ok frame when the kept gap exceeds ``max_gap_sec``."""
    if max_gap_sec <= 0 or not usable:
        return
    for _ in range(len(usable) + 1):
        kept = sorted(
            (frames_by_id[ident] for ident in reasons if ident in frames_by_id),
            key=lambda frame: frame.t_sec,
        )
        if not kept:
            reasons.setdefault(id(usable[0]), []).append("stream_start")
            continue
        times = [frame.t_sec for frame in kept]
        inserted = False
        gaps = list(zip(times, times[1:], strict=False))
        gaps.append((times[-1], None))
        for prev, nxt in gaps:
            limit = prev + max_gap_sec
            if nxt is not None and nxt - prev <= max_gap_sec:
                continue
            if nxt is None:
                window = [
                    frame
                    for frame in usable
                    if frame.t_sec - prev > max_gap_sec and id(frame) not in reasons
                ]
            else:
                window = [
                    frame
                    for frame in usable
                    if prev < frame.t_sec < nxt
                    and frame.t_sec >= limit
                    and id(frame) not in reasons
                ]
                if not window and nxt - prev > max_gap_sec:
                    window = [
                        frame
                        for frame in usable
                        if prev < frame.t_sec < nxt and id(frame) not in reasons
                    ]
            if not window:
                continue
            pick = min(window, key=lambda frame: (abs(frame.t_sec - limit), frame.t_sec))
            bucket = reasons.setdefault(id(pick), [])
            if "max_gap" not in bucket:
                bucket.append("max_gap")
            inserted = True
            break
        if not inserted:
            return


def _coalesce(
    picks: list[KeyframePick], min_spacing_sec: float
) -> tuple[list[KeyframePick], list[SkipSpan]]:
    if min_spacing_sec <= 0 or len(picks) < 2:
        return picks, []
    kept: list[KeyframePick] = [picks[0]]
    skips: list[SkipSpan] = []
    for pick in picks[1:]:
        previous = kept[-1]
        if pick.t_sec - previous.t_sec >= min_spacing_sec:
            kept.append(pick)
            continue
        if pick.forced and not previous.forced:
            skips.append(_point_skip(previous.t_sec, pick.t_sec))
            kept[-1] = pick
            continue
        if previous.forced and not pick.forced:
            skips.append(_point_skip(pick.t_sec, pick.t_sec + min_spacing_sec))
            continue
        if pick.forced and previous.forced:
            kept.append(pick)
            continue
        end = max(pick.t_sec + 1e-3, previous.t_sec + min_spacing_sec)
        skips.append(_point_skip(pick.t_sec, end))
    return kept, skips


def _point_skip(start_sec: float, end_sec: float) -> SkipSpan:
    if end_sec <= start_sec:
        end_sec = start_sec + 1e-3
    return SkipSpan(start_sec=start_sec, end_sec=end_sec, skipped_frames=1, reason="coalesce")


def _quality_skips(frames: list[FrameSignal]) -> list[SkipSpan]:
    spans: list[SkipSpan] = []
    index = 0
    while index < len(frames):
        frame = frames[index]
        if frame.quality_ok or frame.t_sec < 0:
            index += 1
            continue
        start = index
        while index < len(frames) and not frames[index].quality_ok and frames[index].t_sec >= 0:
            index += 1
        group = frames[start:index]
        end = frames[index].t_sec if index < len(frames) else group[-1].t_sec + 1e-3
        if end <= group[0].t_sec:
            end = group[0].t_sec + 1e-3
        spans.append(
            SkipSpan(
                start_sec=group[0].t_sec,
                end_sec=end,
                skipped_frames=len(group),
                reason="quality",
            )
        )
    return spans
