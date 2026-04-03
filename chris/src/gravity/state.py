"""Data structures for the gravity simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class ParticleState:
    """Minimal particle state for the toy gravity model.

    positions and velocities have shape (N, 2) for 2D or (N, 3) for 3D.
    """

    positions: np.ndarray  # shape (N, 2) or (N, 3)
    velocities: np.ndarray  # shape (N, 2) or (N, 3)
    masses: np.ndarray  # shape (N,)


class AccelerationFn(Protocol):
    """Callable type for computing accelerations."""

    def __call__(self, state: ParticleState) -> np.ndarray:  # pragma: no cover
        ...
