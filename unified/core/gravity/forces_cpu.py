"""CPU implementations of gravitational forces (2D and 3D).

Loop and vectorized O(N^2) gravity with softened Newtonian potential.
Works for positions of shape (N, 2) or (N, 3).

Optional GPU path (CuPy/Numba CUDA) for large N: see docs/GPU.md.
"""

from __future__ import annotations

import numpy as np

from .state import ParticleState


def compute_accelerations(
    state: ParticleState,
    softening: float = 0.05,
    G: float = 1.0,
) -> np.ndarray:
    """Compute accelerations for all particles using softened Newtonian gravity.

    Parameters
    ----------
    state:
        Current particle positions, velocities, and masses.
    softening:
        Softening length ε to avoid singular forces at very small separations.
    G:
        Gravitational constant in code units.
    """
    pos = state.positions
    masses = state.masses
    n = pos.shape[0]

    acc = np.zeros_like(pos)

    # Simple O(N^2) pairwise interaction.
    for i in range(n):
        # Vector from particle i to all j
        dx = pos - pos[i]
        # Squared distance with softening
        r2 = np.sum(dx * dx, axis=1) + softening * softening

        # Exclude self-interaction
        r2[i] = 1.0  # temporary safe value

        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        inv_r3[i] = 0.0

        # Sum m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
        acc[i] = G * np.sum(masses[:, None] * dx * inv_r3[:, None], axis=0)

    return acc


def compute_accelerations_vectorized(
    state: ParticleState,
    softening: float = 0.05,
    G: float = 1.0,
) -> np.ndarray:
    """Compute accelerations using vectorized NumPy (broadcasting). Faster for N ~ 1k–5k.

    Same physics as compute_accelerations; use for production runs. Uses O(N^2) memory.
    """
    pos = state.positions
    masses = state.masses
    n = pos.shape[0]

    # dx[i, j] = pos[j] - pos[i]  -> shape (N, N, ndim)
    dx = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(dx * dx, axis=-1) + softening * softening
    np.fill_diagonal(r2, 1.0)  # avoid div by zero for self
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)

    # acc[i] = G * sum_j m_j * dx[i,j] * inv_r3[i,j]
    acc = G * (masses[None, :, None] * dx * inv_r3[:, :, None]).sum(axis=1)
    return acc.astype(pos.dtype)


def compute_halo_acceleration(
    positions: np.ndarray,
    M_halo: float,
    a_halo: float,
    G: float = 1.0,
) -> np.ndarray:
    """Acceleration from a static Hernquist dark-matter halo centred at the origin.

    a(r) = -G * M_halo / (r + a)^2  in the radial direction.
    Hernquist enclosed mass: M(<r) = M_halo * r^2 / (r + a)^2.
    """
    r = np.sqrt(np.sum(positions * positions, axis=-1, keepdims=True))
    r_safe = np.maximum(r, 1e-12)
    acc = -G * M_halo / (r_safe + a_halo) ** 2 * (positions / r_safe)
    return acc.astype(positions.dtype)


# Re-export diagnostics for backward compatibility (canonical implementations in diagnostics.py)
from .diagnostics import (
    compute_kinetic_energy,
    compute_potential_energy,
    compute_total_energy,
)

