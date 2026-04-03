"""Simple procedural tests for the 2D gravity implementation (Phase 1).

Run with:

    cd src
    python -m gravity.tests_2d
"""

from __future__ import annotations

import math

import numpy as np

from .diagnostics import compute_angular_momentum
from .forces_cpu import (
    compute_accelerations,
    compute_accelerations_vectorized,
    compute_total_energy,
)
from .integrators import leapfrog_step
from .state import ParticleState


def _two_body_state(distance: float = 1.0, mass: float = 1.0) -> ParticleState:
    """Construct a simple symmetric two-body configuration on the x-axis."""
    pos = np.array(
        [
            [-0.5 * distance, 0.0],
            [0.5 * distance, 0.0],
        ],
        dtype=float,
    )
    vel = np.zeros_like(pos)
    masses = np.array([mass, mass], dtype=float)
    return ParticleState(positions=pos, velocities=vel, masses=masses)


def test_two_body_symmetry() -> None:
    """Check that accelerations are equal and opposite for two equal masses."""
    state = _two_body_state(distance=1.0, mass=1.0)
    acc = compute_accelerations(state, softening=0.01, G=1.0)

    a0 = acc[0]
    a1 = acc[1]

    print("Two-body accelerations:", a0, a1)

    # They should be opposite vectors and equal in magnitude.
    mag0 = float(np.linalg.norm(a0))
    mag1 = float(np.linalg.norm(a1))

    if mag0 == 0.0 or mag1 == 0.0:
        print("FAIL: one of the accelerations is zero.")
        return

    # Relative magnitude difference
    rel_diff = abs(mag0 - mag1) / max(mag0, mag1)

    # Cosine of angle between them should be close to -1
    cos_angle = float(np.dot(a0, a1) / (mag0 * mag1))

    print(f" |a0|={mag0:.6g}, |a1|={mag1:.6g}, rel_diff={rel_diff:.3e}")
    print(f" cos(angle) between a0 and a1 = {cos_angle:.6g}")

    if rel_diff < 1e-6 and cos_angle < -0.999:
        print("PASS: two-body symmetry looks good.")
    else:
        print("WARN: symmetry check not within tight tolerance.")


def test_center_of_mass_motion(steps: int = 200, dt: float = 0.01) -> None:
    """Check approximate conservation of center-of-mass velocity."""
    # Three equal masses at vertices of an equilateral triangle.
    R = 1.0
    pos = np.array(
        [
            [0.0, R],
            [-0.5 * R, -math.sqrt(3) * R / 2.0],
            [0.5 * R, -math.sqrt(3) * R / 2.0],
        ],
        dtype=float,
    )
    vel = np.zeros_like(pos)
    masses = np.ones(3, dtype=float)
    state = ParticleState(positions=pos, velocities=vel, masses=masses)

    def accel_fn(s: ParticleState) -> np.ndarray:
        return compute_accelerations(s, softening=0.05, G=1.0)

    # Initial COM quantities
    m_tot = float(np.sum(masses))
    com_pos0 = np.sum(masses[:, None] * state.positions, axis=0) / m_tot
    com_vel0 = np.sum(masses[:, None] * state.velocities, axis=0) / m_tot

    com_pos = com_pos0.copy()
    com_vel = com_vel0.copy()

    for _ in range(steps):
        state = leapfrog_step(state, dt=dt, accel_fn=accel_fn)
        com_pos = np.sum(masses[:, None] * state.positions, axis=0) / m_tot
        com_vel = np.sum(masses[:, None] * state.velocities, axis=0) / m_tot

    print("Center-of-mass position (initial, final):", com_pos0, com_pos)
    print("Center-of-mass velocity (initial, final):", com_vel0, com_vel)

    pos_shift = float(np.linalg.norm(com_pos - com_pos0))
    vel_shift = float(np.linalg.norm(com_vel - com_vel0))

    print(f" COM position shift = {pos_shift:.3e}")
    print(f" COM velocity shift = {vel_shift:.3e}")

    # These thresholds are intentionally loose and can be tightened later.
    if pos_shift < 1e-2 and vel_shift < 1e-3:
        print("PASS: COM motion is reasonably conserved for this test.")
    else:
        print("WARN: COM drift is non-negligible; consider smaller dt or tweaks.")


def test_energy_drift(steps: int = 1000, dt: float = 0.001) -> None:
    """Roughly monitor total energy drift for a simple two-body orbit."""
    # Two-body orbit: one heavy, one light, roughly circular.
    M = 10.0
    m = 1.0
    distance = 1.0

    # Positions along x-axis (COM at ~0)
    pos = np.array(
        [
            [-m / (M + m) * distance, 0.0],
            [M / (M + m) * distance, 0.0],
        ],
        dtype=float,
    )

    # Approximate circular orbital speed for reduced mass system (toy)
    G = 1.0
    v = math.sqrt(G * (M + m) / distance)

    vel = np.array(
        [
            [0.0, -m / (M + m) * v],
            [0.0, M / (M + m) * v],
        ],
        dtype=float,
    )
    masses = np.array([M, m], dtype=float)

    state = ParticleState(positions=pos, velocities=vel, masses=masses)

    def accel_fn(s: ParticleState) -> np.ndarray:
        return compute_accelerations(s, softening=0.05, G=G)

    E0 = compute_total_energy(state, softening=0.05, G=G)
    Emin = E0
    Emax = E0

    for _ in range(steps):
        state = leapfrog_step(state, dt=dt, accel_fn=accel_fn)
        E = compute_total_energy(state, softening=0.05, G=G)
        Emin = min(Emin, E)
        Emax = max(Emax, E)

    print(f"Initial total energy E0 = {E0:.6g}")
    print(f"Min(E), Max(E) over run = {Emin:.6g}, {Emax:.6g}")

    rel_drift = max(abs(Emax - E0), abs(Emin - E0)) / max(1.0, abs(E0))
    print(f"Relative energy drift ≈ {rel_drift:.3e}")

    # Again, this is a loose threshold to start with.
    if rel_drift < 1e-2:
        print("PASS: energy drift is reasonably small for this test.")
    else:
        print("WARN: noticeable energy drift; try smaller dt or review forces.")


def test_circular_orbit_stability(
    steps: int = 500,
    dt: float = 0.005,
    r0: float = 1.0,
    M_star: float = 10.0,
    m_particle: float = 0.001,
    G: float = 1.0,
    softening: float = 0.01,
) -> None:
    """Single particle in circular orbit around central mass; radius should stay roughly constant."""
    v_circ = math.sqrt(G * M_star / r0)
    pos = np.array(
        [
            [0.0, 0.0],
            [r0, 0.0],
        ],
        dtype=float,
    )
    vel = np.array(
        [
            [0.0, 0.0],
            [0.0, v_circ],
        ],
        dtype=float,
    )
    masses = np.array([M_star, m_particle], dtype=float)
    state = ParticleState(positions=pos, velocities=vel, masses=masses)

    def accel_fn(s: ParticleState) -> np.ndarray:
        return compute_accelerations_vectorized(
            s, softening=softening, G=G
        )

    for _ in range(steps):
        state = leapfrog_step(state, dt=dt, accel_fn=accel_fn)

    r_final = float(np.linalg.norm(state.positions[1] - state.positions[0]))
    rel_err = abs(r_final - r0) / r0
    print(f"  Initial radius = {r0:.6g}, final radius = {r_final:.6g}, rel_err = {rel_err:.3e}")

    if rel_err < 0.1:
        print("PASS: circular orbit radius is reasonably stable.")
    else:
        print("WARN: orbit radius drifted; try smaller dt or check forces.")


def test_vectorized_vs_loop_forces() -> None:
    """Vectorized and loop acceleration implementations should agree."""
    from .init_conditions import make_disk_2d

    state = make_disk_2d(20, seed=123, M_star=1.0, r_min=0.5, r_max=1.5)
    acc_loop = compute_accelerations(state, softening=0.05, G=1.0)
    acc_vec = compute_accelerations_vectorized(state, softening=0.05, G=1.0)

    diff = np.abs(acc_loop - acc_vec)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    print(f"  max |a_loop - a_vec| = {max_diff:.3e}, mean = {mean_diff:.3e}")

    if max_diff < 1e-10:
        print("PASS: vectorized and loop forces match.")
    else:
        print("WARN: vectorized and loop forces differ.")


def test_angular_momentum_conservation(
    steps: int = 300,
    dt: float = 0.01,
) -> None:
    """Total angular momentum should be conserved (2D, Lz only)."""
    from .init_conditions import make_disk_2d

    state = make_disk_2d(50, seed=456, M_star=1.0, r_min=0.5, r_max=2.0)

    def accel_fn(s: ParticleState) -> np.ndarray:
        return compute_accelerations_vectorized(s, softening=0.05, G=1.0)

    L0 = compute_angular_momentum(state)
    Lmin, Lmax = L0, L0

    for _ in range(steps):
        state = leapfrog_step(state, dt=dt, accel_fn=accel_fn)
        L = compute_angular_momentum(state)
        Lmin = min(Lmin, L)
        Lmax = max(Lmax, L)

    rel_drift = max(abs(Lmax - L0), abs(Lmin - L0)) / max(1e-20, abs(L0))
    print(f"  Initial Lz = {L0:.6g}, min = {Lmin:.6g}, max = {Lmax:.6g}")
    print(f"  Relative Lz drift = {rel_drift:.3e}")

    if rel_drift < 1e-2:
        print("PASS: angular momentum is reasonably conserved.")
    else:
        print("WARN: angular momentum drift is non-negligible.")


def run_all_tests() -> None:
    """Run all simple procedural tests."""
    print("=== Test: two-body symmetry ===")
    test_two_body_symmetry()
    print()

    print("=== Test: COM motion ===")
    test_center_of_mass_motion()
    print()

    print("=== Test: energy drift ===")
    test_energy_drift()
    print()

    print("=== Test: circular orbit stability ===")
    test_circular_orbit_stability()
    print()

    print("=== Test: vectorized vs loop forces ===")
    test_vectorized_vs_loop_forces()
    print()

    print("=== Test: angular momentum conservation ===")
    test_angular_momentum_conservation()


if __name__ == "__main__":
    run_all_tests()
