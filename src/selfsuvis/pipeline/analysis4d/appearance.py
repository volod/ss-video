"""Versioned masked appearance descriptors and time-decayed track prototypes.

Two vectors associate only when model id, revision, dimension, and
preprocessing version match. A mismatched space is refused rather than scored.
"""

import math
from dataclasses import dataclass

import numpy as np

PREPROCESS = "analysis4d-masked-dino-v1"


class IncompatibleEmbedding(ValueError):
    """The two descriptors do not live in one versioned space."""


@dataclass(frozen=True)
class EmbeddingSpace:
    """Identity of one descriptor space. Any difference blocks association."""

    model_id: str
    revision: str
    dim: int
    preprocessing_version: str = PREPROCESS

    def compatible(self, other: "EmbeddingSpace") -> bool:
        return (
            self.model_id == other.model_id
            and self.revision == other.revision
            and self.dim == other.dim
            and self.preprocessing_version == other.preprocessing_version
        )


@dataclass(frozen=True)
class Descriptor:
    """One L2-normalized vector in ``space``."""

    space: EmbeddingSpace
    vector: np.ndarray
    t_sec: float = 0.0

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float64).reshape(-1)
        if vector.shape != (self.space.dim,):
            raise ValueError("descriptor length does not match the embedding space")
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        object.__setattr__(self, "vector", vector)


def pool_masked(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean-pool ``features`` (H, W, D) over the true pixels of ``mask``."""
    grid = np.asarray(features, dtype=np.float64)
    keep = np.asarray(mask, dtype=bool)
    if grid.ndim != 3:
        raise ValueError("features must have shape (H, W, D)")
    if keep.shape != grid.shape[:2]:
        keep = _resize_mask(keep, grid.shape[1], grid.shape[0])
    selected = grid[keep]
    if selected.size == 0:
        selected = grid.reshape(-1, grid.shape[-1])
    pooled = selected.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled


def cosine_association(left: Descriptor, right: Descriptor) -> float:
    """Cosine of two descriptors. Incompatible versions raise."""
    if not left.space.compatible(right.space):
        raise IncompatibleEmbedding(
            f"{left.space.model_id}@{left.space.revision} dim {left.space.dim} "
            f"cannot associate with {right.space.model_id}@{right.space.revision} "
            f"dim {right.space.dim}"
        )
    return float(np.dot(left.vector, right.vector))


def decay_prototype(
    previous: Descriptor, observation: Descriptor, *, tau_sec: float = 2.0
) -> Descriptor:
    """Blend ``observation`` into ``previous`` with exponential time decay.

    Args:
        previous: Current track prototype.
        observation: New masked descriptor. Must share ``previous.space``.
        tau_sec: Decay constant. Older prototypes keep less weight.

    Returns:
        The updated prototype at the observation time.
    """
    if not previous.space.compatible(observation.space):
        raise IncompatibleEmbedding("track prototype space does not match the observation")
    dt = max(0.0, observation.t_sec - previous.t_sec)
    tau = max(float(tau_sec), 1e-6)
    weight = math.exp(-dt / tau)
    mixed = weight * previous.vector + (1.0 - weight) * observation.vector
    return Descriptor(previous.space, mixed, t_sec=observation.t_sec)


class PrototypeBank:
    """Per-track prototypes. A new model revision does not update the old one."""

    def __init__(self, tau_sec: float = 2.0):
        self.tau_sec = tau_sec
        self._rows: dict[str, Descriptor] = {}

    def update(self, track_id: str, observation: Descriptor) -> Descriptor:
        """Insert or decay-update ``track_id`` and return the prototype."""
        current = self._rows.get(track_id)
        if current is None:
            stored = Descriptor(observation.space, observation.vector, t_sec=observation.t_sec)
        else:
            stored = decay_prototype(current, observation, tau_sec=self.tau_sec)
        self._rows[track_id] = stored
        return stored

    def get(self, track_id: str) -> Descriptor | None:
        return self._rows.get(track_id)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.size == 0 or width < 1 or height < 1:
        return np.zeros((height, width), dtype=bool)
    rows = np.linspace(0, mask.shape[0] - 1, height).astype(int)
    cols = np.linspace(0, mask.shape[1] - 1, width).astype(int)
    return mask[rows][:, cols]
