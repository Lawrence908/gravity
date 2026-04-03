"""Simple procedural tests for the 3D gravity implementation."""

from __future__ import annotations

import math

import numpy as np

from .diagnostics import (
    compute_angular_momentum,
    compute_angular_momentum_vector,
    compute_total_energy,
)
from .forces_cpu import compute_accelerations_vectorized
from .init_conditions import make_disk_3d
from .integrators import leapfrog_step
from .state import ParticleState


def test_energy_drift_3d(steps: int = 500, dt: float = 0.005) -> None:
    """Total energy should be approximately conserved in 3D disk run."""
    state = make_disk_3d(30, seed=42, M_star=1.0, r_min=0.5, r_max=1.5, thickness=0.05)

    def accel_fn(s: ParticleState) -> np.ndarray:
        return compute_accelerations_vectorized(s, softening=0.05, G=1.0)

    E0 = compute_total_energy(state, softening=0.05, G=1.0)
    for _ in range(steps):
        state = leapfrog_step(state, dt=dt, accel_fn=accel_fn)

    E_final = compute_total_energy(state, softening=0.05, G=1.0)
    rel_drift = abs(E_final - E0) / max(1.0, abs(E0))
    print(f"  3D energy: E0={E0:.6g}, E_final={E_final:.6g}, rel_drift={rel_drift:.3e}")

    assert rel_drift < 0.05, f"3D energy drift {rel_drift:.4g} too large"
    print("PASS: 3D energy drift test.")


def test_circular_orbit_3d(steps: int = 200, dt: float = 0.01, r0: float = 1.0) -> None:
    """Single particle in circular orbit in xy-plane (z=0); radius should stay roughly constant."""
    M_star = 10.0
    m_particle = 0.001
    G = 1.0
    v_circ = math.sqrt(G * M_star / r0)

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [r0, 0.0, 0.0],
        ],
        dtype=float,
    )
    vel = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, v_circ, 0.0],
        ],
        dtype=float,
    )
    masses = np.array([M_star, m_particle], dtype=float)
    state = ParticleState(positions=pos, velocities=vel, masses=masses)

    def accel_fn(s: ParticleState) -> np.ndarray:
        return compute_accelerations_vectorized(s, softening=0.01, G=G)

    for _ in range(steps):
        state = leapfrog_step(state, dt=dt, accel_fn=accel_fn)

    r_final = float(np.linalg.norm(state.positions[1] - state.positions[0]))
    rel_err = abs(r_final - r0) / r0
    print(f"  3D circular orbit: r0={r0:.4g}, r_final={r_final:.4g}, rel_err={rel_err:.3e}")

    assert rel_err < 0.15, f"3D orbit radius drift {rel_err:.4g} too large"
    print("PASS: 3D circular orbit stability.")


def test_angular_momentum_conservation_3d(steps: int = 300, dt: float = 0.01) -> None:
    """Total angular momentum vector should be conserved in 3D."""
    state = make_disk_3d(40, seed=99, M_star=1.0, r_min=0.5, r_max=2.0, thickness=0.08)

    def accel_fn(s: ParticleState) -> np.ndarray:
        return compute_accelerations_vectorized(s, softening=0.05, G=1.0)

    L0 = compute_angular_momentum_vector(state)
    Lz0 = compute_angular_momentum(state)

    for _ in range(steps):
        state = leapfrog_step(state, dt=dt, accel_fn=accel_fn)

    L_final = compute_angular_momentum_vector(state)
    Lz_final = compute_angular_momentum(state)
    drift = np.linalg.norm(L_final - L0)
    rel_drift = drift / max(1e-20, np.linalg.norm(L0))
    print(f"  3D L vector: |L0|={np.linalg.norm(L0):.6g}, |L_final|={np.linalg.norm(L_final):.6g}")
    print(f"  Lz: {Lz0:.6g} -> {Lz_final:.6g}, rel_drift={rel_drift:.3e}")

    assert rel_drift < 0.05, f"3D angular momentum drift {rel_drift:.4g} too large"
    print("PASS: 3D angular momentum conservation.")


def run_all_tests() -> None:
    print("=== Test: 3D energy drift ===")
    test_energy_drift_3d()
    print()

    print("=== Test: 3D circular orbit ===")
    test_circular_orbit_3d()
    print()

    print("=== Test: 3D angular momentum conservation ===")
    test_angular_momentum_conservation_3d()


if __name__ == "__main__":
    run_all_tests()
