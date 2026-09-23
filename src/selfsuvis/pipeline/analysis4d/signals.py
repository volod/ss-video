"""Build frame signals from image files.

The selector compares a small grayscale thumbnail. The full image stays on
disk and is opened only when a grounding call needs it.
"""

from pathlib import Path

from PIL import Image

from selfsuvis.pipeline.analysis4d.keyframes import FrameSignal


def load_rgb(source: object) -> Image.Image | None:
    """Return an RGB image from a path or an existing image.

    Args:
        source: Filesystem path or an object with ``convert``. ``None`` and a
            missing file return ``None``.

    Returns:
        A detached RGB image, or ``None`` when the source cannot be read.
    """
    if source is None:
        return None
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    path = Path(str(source))
    if not path.is_file():
        return None
    try:
        with Image.open(path) as handle:
            return handle.convert("RGB")
    except OSError:
        return None


def appearance_embedding(image: Image.Image) -> tuple[float, ...]:
    """Return an 8x8 grayscale vector in ``[0, 1]`` plus a constant bias.

    The bias keeps a uniform exposure change from looking like the same
    direction to the keyframe selector.
    """
    gray = image.convert("L").resize((8, 8))
    return tuple(pixel / 255.0 for pixel in gray.getdata()) + (1.0,)


def cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return one minus the cosine similarity, or 0 when either vector is empty."""
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(width))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return 1.0 - cosine


def signal_drift(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return the larger of cosine distance and mean absolute difference."""
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    mean_abs = sum(abs(left[index] - right[index]) for index in range(width)) / width
    return max(cosine_distance(left, right), mean_abs)


def signals_from_paths(samples: list[tuple[float, str]]) -> list[FrameSignal]:
    """Build ordered signals. A missing image is not a usable frame.

    Args:
        samples: ``(t_sec, path)`` pairs. Paths are stored on the signal so the
            full raster is not kept in memory.

    Returns:
        Signals sorted by timestamp. Drift is measured from the first readable
        frame so the keyframe gate can see a scene change.
    """
    loaded: list[tuple[float, str, tuple[float, ...]]] = []
    for t_sec, path in samples:
        image = load_rgb(path)
        if image is None:
            loaded.append((t_sec, path, ()))
            continue
        loaded.append((t_sec, path, appearance_embedding(image)))
    first = next((embedding for _, _, embedding in loaded if embedding), ())
    frames = [
        FrameSignal(
            t_sec=t_sec,
            embedding=embedding or (1.0, 0.0),
            drift=signal_drift(first, embedding) if embedding else 0.0,
            image=path if embedding else None,
            quality_ok=bool(embedding),
        )
        for t_sec, path, embedding in loaded
    ]
    frames.sort(key=lambda item: item.t_sec)
    return frames
