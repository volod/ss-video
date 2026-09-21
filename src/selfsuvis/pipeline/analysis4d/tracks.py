"""Track lifecycle for prompted 2D instances.

Association uses box IoU, constant-velocity prediction, label compatibility,
and appearance cosine. A long gap or a hard cut ends the track. The deep
profile may add an identity link to the earlier track. It does not reuse the
id and it does not rewrite the earlier observation. Count-only detections are
ignored here.
"""

from dataclasses import dataclass, field

from selfsuvis.pipeline.analysis4d.geometry import box_iou_2d
from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal

_ACTIVE = frozenset({"tentative", "confirmed", "occluded"})


@dataclass(frozen=True)
class Detection:
    """One grounding hit. ``count_only`` hits are not identities."""

    prompt_id: str
    label_raw: str
    label_normalized: str
    xywh: tuple[float, float, float, float]
    confidence: float
    appearance: tuple[float, ...] = ()
    mask_bits: str | None = None
    negative_prompts: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = ()
    count_only: bool = False


@dataclass
class Observation:
    """One emitted track sample before it is written as a contract record."""

    track_id: str
    state: str
    t_sec: float
    prompt_id: str
    label_raw: str
    label_normalized: str
    xywh: tuple[float, float, float, float]
    confidence: float
    appearance: tuple[float, ...] = ()
    mask_bits: str | None = None
    negative_prompts: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = ()
    identity_link_id: str | None = None
    identity_reason: str | None = None


@dataclass(frozen=True)
class TrackerConfig:
    """Match, occlusion, and re-identification thresholds."""

    match_min: float = 0.40
    low_association: float = 0.62
    occluded_sec: float = 1.0
    lost_sec: float = 2.0
    reid_cosine: float = 0.85
    seed: int = 0


@dataclass
class MemoryReset:
    """In-memory reset note. The pass writer copies it into the audit."""

    t_sec: float
    reason: str
    scope: str


@dataclass
class _Live:
    track_id: str
    state: str
    hits: int
    last_match_t: float
    t_sec: float
    box: tuple[float, float, float, float]
    velocity: tuple[float, float]
    label_raw: str
    label_normalized: str
    prompt_id: str
    appearance: tuple[float, ...]
    confidence: float
    negative_prompts: tuple[str, ...]
    exemplars: tuple[str, ...]
    mask_bits: str | None = None
    closed: bool = False


@dataclass
class Tracker:
    """Causal track state. Call :meth:`link_identities` after the deep pass."""

    mission_id: str
    profile: str
    config: TrackerConfig = field(default_factory=TrackerConfig)
    id_prefix: str = ""
    observations: list[Observation] = field(default_factory=list)
    resets: list[MemoryReset] = field(default_factory=list)
    _live: list[_Live] = field(default_factory=list)
    _next: int = 1
    _epoch: int = 1

    def step(self, frame: FrameSignal, detections: list[Detection]) -> list[str]:
        """Advance every live track to ``frame`` and return keyframe reasons."""
        reasons: list[str] = []
        if frame.scene_cut or frame.calibration_changed:
            reason = "scene_cut" if frame.scene_cut else "calibration_change"
            if self._live:
                self._end_all(frame.t_sec)
                reasons.append("track_death")
            self._epoch += 1
            self.resets.append(
                MemoryReset(
                    t_sec=frame.t_sec,
                    reason=reason,
                    scope=f"{self.mission_id}:default:epoch-{self._epoch}",
                )
            )
        self._retire_stale(frame.t_sec, reasons)
        usable = [item for item in detections if not item.count_only]
        self._match(frame, usable, reasons)
        return reasons

    def count_active(self, label: str) -> int:
        """Unique live tracks with ``label`` that a count expert can disagree with."""
        return sum(
            1
            for live in self._live
            if live.label_normalized == label and live.state in _ACTIVE and not live.closed
        )

    def link_identities(self) -> None:
        """Deep profile only: link a new track to an ended one without reusing the id."""
        if self.profile != "deep":
            return
        first: dict[str, Observation] = {}
        ended: dict[str, Observation] = {}
        for observation in self.observations:
            current = first.get(observation.track_id)
            if current is None or observation.t_sec < current.t_sec:
                first[observation.track_id] = observation
            if observation.state == "ended":
                ended[observation.track_id] = observation
        reset_times = [item.t_sec for item in self.resets]
        for track_id, birth in first.items():
            if birth.identity_link_id is not None:
                continue
            best_id = None
            best_end = -1.0
            best_reason = "occlusion"
            for other_id, end in ended.items():
                if other_id == track_id or end.t_sec > birth.t_sec:
                    continue
                if end.label_normalized != birth.label_normalized:
                    continue
                if _cosine(end.appearance, birth.appearance) < self.config.reid_cosine:
                    continue
                cut = any(end.t_sec < stamp <= birth.t_sec for stamp in reset_times)
                if birth.t_sec - end.t_sec < self.config.lost_sec and not cut:
                    continue
                if end.t_sec >= best_end:
                    best_end = end.t_sec
                    best_id = other_id
                    best_reason = "camera_cut" if cut else "occlusion"
            if best_id is not None:
                birth.identity_link_id = best_id
                birth.identity_reason = best_reason

    def _retire_stale(self, t_sec: float, reasons: list[str]) -> None:
        staying: list[_Live] = []
        for live in self._live:
            if t_sec - live.last_match_t > self.config.lost_sec:
                self._emit_closed(live, t_sec, "ended")
                reasons.append("track_death")
            else:
                staying.append(live)
        self._live = staying

    def _end_all(self, t_sec: float) -> None:
        for live in self._live:
            self._emit_closed(live, t_sec, "ended")
        self._live = []

    def _match(self, frame: FrameSignal, detections: list[Detection], reasons: list[str]) -> None:
        existing = list(self._live)
        predicted = {live.track_id: _predict(live, frame.t_sec) for live in existing}
        pairs: list[tuple[float, str, int]] = []
        for live in self._live:
            for index, detection in enumerate(detections):
                if live.label_normalized != detection.label_normalized:
                    continue
                score = _score(live, predicted[live.track_id], detection)
                if score >= self.config.match_min:
                    pairs.append((score, live.track_id, index))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2], _tie(self.config.seed, item[2])))
        used_tracks: set[str] = set()
        used_dets: set[int] = set()
        matches: list[tuple[float, str, int]] = []
        for score, track_id, index in pairs:
            if track_id in used_tracks or index in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(index)
            matches.append((score, track_id, index))
        live_by_id = {live.track_id: live for live in self._live}
        for score, track_id, index in matches:
            if score < self.config.low_association:
                reasons.append("low_association")
            self._assign(live_by_id[track_id], detections[index], frame.t_sec)
        for index, detection in enumerate(detections):
            if index in used_dets:
                continue
            self._birth(detection, frame.t_sec)
            reasons.append("track_birth")
        for live in existing:
            if live.track_id in used_tracks or live.closed:
                continue
            gap = frame.t_sec - live.last_match_t
            box = predicted[live.track_id]
            if gap <= self.config.occluded_sec:
                self._emit(live, frame.t_sec, "occluded", box, live.confidence)
            elif live.state != "lost":
                self._emit(live, frame.t_sec, "lost", box, live.confidence)

    def _assign(self, live: _Live, detection: Detection, t_sec: float) -> None:
        old_cx = live.box[0] + live.box[2] / 2.0
        old_cy = live.box[1] + live.box[3] / 2.0
        box = _clamp(detection.xywh)
        dt = max(1e-3, t_sec - live.t_sec)
        live.velocity = (
            (box[0] + box[2] / 2.0 - old_cx) / dt,
            (box[1] + box[3] / 2.0 - old_cy) / dt,
        )
        live.box = box
        live.t_sec = t_sec
        live.last_match_t = t_sec
        live.hits += 1
        live.state = "confirmed" if live.hits >= 2 else "tentative"
        live.confidence = detection.confidence
        live.appearance = detection.appearance or live.appearance
        live.mask_bits = detection.mask_bits
        live.label_raw = detection.label_raw
        live.negative_prompts = detection.negative_prompts
        live.exemplars = detection.exemplars
        self._emit(live, t_sec, live.state, box, detection.confidence, detection.mask_bits)

    def _birth(self, detection: Detection, t_sec: float) -> None:
        track_id = f"{self.id_prefix}track-{self._next:04d}"
        self._next += 1
        box = _clamp(detection.xywh)
        live = _Live(
            track_id=track_id,
            state="tentative",
            hits=1,
            last_match_t=t_sec,
            t_sec=t_sec,
            box=box,
            velocity=(0.0, 0.0),
            label_raw=detection.label_raw,
            label_normalized=detection.label_normalized,
            prompt_id=detection.prompt_id,
            appearance=detection.appearance,
            confidence=detection.confidence,
            negative_prompts=detection.negative_prompts,
            exemplars=detection.exemplars,
            mask_bits=detection.mask_bits,
        )
        self._live.append(live)
        self._emit(live, t_sec, "tentative", box, detection.confidence, detection.mask_bits)

    def _emit_closed(self, live: _Live, t_sec: float, state: str) -> None:
        live.closed = True
        live.state = state
        self._emit(live, t_sec, state, live.box, live.confidence, live.mask_bits)

    def _emit(
        self,
        live: _Live,
        t_sec: float,
        state: str,
        box: tuple[float, float, float, float],
        confidence: float,
        mask_bits: str | None = None,
    ) -> None:
        live.state = state
        live.t_sec = t_sec
        self.observations.append(
            Observation(
                track_id=live.track_id,
                state=state,
                t_sec=t_sec,
                prompt_id=live.prompt_id,
                label_raw=live.label_raw,
                label_normalized=live.label_normalized,
                xywh=_clamp(box),
                confidence=min(1.0, max(0.0, confidence)),
                appearance=live.appearance,
                mask_bits=mask_bits if mask_bits is not None else live.mask_bits,
                negative_prompts=live.negative_prompts,
                exemplars=live.exemplars,
            )
        )


def _predict(live: _Live, t_sec: float) -> tuple[float, float, float, float]:
    dt = t_sec - live.t_sec
    cx = live.box[0] + live.box[2] / 2.0 + live.velocity[0] * dt
    cy = live.box[1] + live.box[3] / 2.0 + live.velocity[1] * dt
    return _clamp((cx - live.box[2] / 2.0, cy - live.box[3] / 2.0, live.box[2], live.box[3]))


def _score(
    live: _Live, predicted: tuple[float, float, float, float], detection: Detection
) -> float:
    iou = box_iou_2d(list(predicted), list(detection.xywh))
    motion = max(0.0, 1.0 - _center_distance(predicted, detection.xywh) / 0.5)
    appearance = max(0.0, _cosine(live.appearance, detection.appearance))
    return 0.45 * iou + 0.25 * motion + 0.15 * 1.0 + 0.15 * appearance


def _center_distance(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    dx = (left[0] + left[2] / 2.0) - (right[0] + right[2] / 2.0)
    dy = (left[1] + left[3] / 2.0) - (right[1] + right[3] / 2.0)
    return (dx * dx + dy * dy) ** 0.5


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return dot / (left_norm * right_norm)


def _clamp(xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    _x, _y, width, height = xywh
    width = min(1.0, max(1e-3, float(width)))
    height = min(1.0, max(1e-3, float(height)))
    x = min(max(0.0, float(_x)), 1.0 - width)
    y = min(max(0.0, float(_y)), 1.0 - height)
    x = round(x, 6)
    y = round(y, 6)
    width = round(width, 6)
    height = round(height, 6)
    if x + width > 1:
        x = round(1.0 - width, 6)
    if y + height > 1:
        y = round(1.0 - height, 6)
    return (x, y, width, height)


def _tie(seed: int, index: int) -> int:
    return ((seed + 1) * (index + 1)) % 1_000_003
