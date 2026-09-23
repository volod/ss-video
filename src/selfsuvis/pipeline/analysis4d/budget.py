"""Bounded frame queue and GPU admission for 4D profiles.

Track updates are required. Optional stages (dense geometry, VLM, review) are
shed when the queue had to coalesce or when no GPU slot is free. Discarded
frames become gap spans. Timestamps are never reordered.
"""

from dataclasses import dataclass, field

from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal

OPTIONAL_STAGES = ("dense_geometry", "vlm", "review")
TRACK_STAGE = "tracks"


@dataclass(frozen=True)
class DiscardedSpan:
    """One discarded frame interval. ``dropped_t_sec`` lies in ``[start_sec, end_sec)``."""

    start_sec: float
    end_sec: float
    skipped_frames: int
    reason: str
    dropped_t_sec: float


@dataclass
class BoundedFrameQueue:
    """In-order queue that coalesces redundant frames once it is full."""

    capacity: int
    _items: list[FrameSignal] = field(default_factory=list)
    max_depth: int = 0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("queue capacity must be at least 1")

    @property
    def depth(self) -> int:
        return len(self._items)

    def push(self, frame: FrameSignal) -> list[DiscardedSpan]:
        """Append ``frame`` and coalesce until the depth is within capacity.

        Args:
            frame: Next decoded frame. Its timestamp must not precede the queue.

        Returns:
            Spans for frames discarded by this call. Empty when the frame fit.

        Raises:
            ValueError: ``frame`` is earlier than the newest queued frame.
        """
        if self._items and frame.t_sec < self._items[-1].t_sec:
            raise ValueError("frame timestamps must not go backwards")
        self._items.append(frame)
        spans: list[DiscardedSpan] = []
        while len(self._items) > self.capacity:
            spans.append(self._coalesce())
        self.max_depth = max(self.max_depth, len(self._items))
        return spans

    def drain(self) -> list[FrameSignal]:
        """Return queued frames in arrival order and leave the queue empty."""
        items = list(self._items)
        self._items.clear()
        return items

    def _coalesce(self) -> DiscardedSpan:
        index = self._victim()
        dropped = self._items.pop(index)
        following = self._items[index].t_sec if index < len(self._items) else None
        start, end = _interval(dropped.t_sec, following)
        return DiscardedSpan(
            start_sec=start,
            end_sec=end,
            skipped_frames=1,
            reason="queue_coalesce",
            dropped_t_sec=dropped.t_sec,
        )

    def _victim(self) -> int:
        """Prefer a redundant frame over a scene cut, and keep the newest frame."""
        best: int | None = None
        best_gap: float | None = None
        last = len(self._items) - 1
        for index, frame in enumerate(self._items):
            if index == 0 or index == last or _protected(frame):
                continue
            spacing = frame.t_sec - self._items[index - 1].t_sec
            if best is None or best_gap is None or spacing < best_gap:
                best = index
                best_gap = spacing
        if best is not None:
            return best
        for index, frame in enumerate(self._items):
            if not _protected(frame):
                return index
        return 0


@dataclass
class BudgetController:
    """Admit optional GPU stages without delaying track continuity."""

    gpu_slots: int
    _in_use: int = 0
    shed: list[str] = field(default_factory=list)

    def admit(self, stage: str, *, pressure: bool = False) -> bool:
        """Return whether ``stage`` may run.

        Args:
            stage: ``tracks`` always runs. Optional stages need a free slot.
            pressure: When true, optional stages are shed so the queue can drain.
        """
        if stage == TRACK_STAGE:
            return True
        if stage not in OPTIONAL_STAGES:
            return True
        if pressure or self._in_use >= max(0, self.gpu_slots):
            if stage not in self.shed:
                self.shed.append(stage)
            return False
        self._in_use += 1
        return True

    def release(self, stage: str) -> None:
        """Return a slot taken by ``admit``."""
        if stage in OPTIONAL_STAGES and self._in_use:
            self._in_use -= 1


def _protected(frame: FrameSignal) -> bool:
    return frame.scene_cut or frame.calibration_changed or frame.prompt_changed


def _interval(dropped: float, following: float | None) -> tuple[float, float]:
    if following is not None and following > dropped:
        return dropped, following
    return dropped, dropped + 1e-3
