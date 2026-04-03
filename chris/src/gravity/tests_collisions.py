"""Tests for collision resolution (inelastic mergers).

Run with:

    cd src
    python -m gravity.tests_collisions
"""

from __future__ import annotations

import numpy as np

from .collisions import resolve_collisions
from .state import ParticleState


def test_particle_particle_merge() -> None:
    """Two disk particles within r_collide merge into one; mass and momentum conserved."""
    # Star at origin (index 0), two particles very close (indices 1, 2)
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0 + 0.01, 0.0],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.0, -0.05],
        ],
        dtype=float,
    )
    masses = np.array([10.0, 0.5, 0.5], dtype=float)
    state = ParticleState(positions=positions, velocities=velocities, masses=masses)

    r_collide = 0.05  # larger than 0.01 separation
    out = resolve_collisions(state, r_collide, star_index=0)

    # Should have star + one merged particle (N=2)
    assert out.positions.shape[0] == 2
    assert out.masses.shape[0] == 2
    assert out.velocities.shape[0] == 2

    # Star unchanged
    np.testing.assert_array_almost_equal(out.positions[0], [0.0, 0.0])
    np.testing.assert_array_almost_equal(out.velocities[0], [0.0, 0.0])
    assert out.masses[0] == 10.0

    # Merged particle: mass = 0.5 + 0.5 = 1.0
    assert out.masses[1] == 1.0
    # COM position
    np.testing.assert_array_almost_equal(out.positions[1], [1.005, 0.0])
    # Momentum-conserving velocity: (0.5*[0,0.1] + 0.5*[0,-0.05]) / 1.0 = [0, 0.025]
    np.testing.assert_array_almost_equal(out.velocities[1], [0.0, 0.025])


def test_star_accretion() -> None:
    """Particle within r_collide of star is absorbed; star mass increases, N decreases."""
    positions = np.array(
        [
            [0.0, 0.0],
            [0.02, 0.0],
        ],
        dtype=float,
    )
    velocities = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    masses = np.array([1.0, 0.1], dtype=float)
    state = ParticleState(positions=positions, velocities=velocities, masses=masses)

    r_collide = 0.05
    out = resolve_collisions(state, r_collide, star_index=0)

    # Only star remains (N=1)
    assert out.positions.shape[0] == 1
    assert out.masses[0] == 1.0 + 0.1


def test_no_collision_large_r() -> None:
    """When no pair is within r_collide, state is unchanged."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    velocities = np.zeros_like(positions)
    masses = np.array([1.0, 0.1], dtype=float)
    state = ParticleState(positions=positions, velocities=velocities, masses=masses)

    out = resolve_collisions(state, r_collide=0.01, star_index=0)

    assert out.positions.shape[0] == 2
    np.testing.assert_array_almost_equal(out.positions, state.positions)
    np.testing.assert_array_almost_equal(out.masses, state.masses)


def test_resolve_collisions_3d() -> None:
    """Collision resolution works in 3D (star + one particle, particle merges into star)."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
        ],
        dtype=float,
    )
    velocities = np.zeros_like(positions)
    masses = np.array([2.0, 0.2], dtype=float)
    state = ParticleState(positions=positions, velocities=velocities, masses=masses)

    out = resolve_collisions(state, r_collide=0.05, star_index=0)

    assert out.positions.shape == (1, 3)
    assert out.masses[0] == 2.2


def run_all_tests() -> None:
    test_no_collision_large_r()
    print("test_no_collision_large_r passed")
    test_star_accretion()
    print("test_star_accretion passed")
    test_particle_particle_merge()
    print("test_particle_particle_merge passed")
    test_resolve_collisions_3d()
    print("test_resolve_collisions_3d passed")
    print("All collision tests passed.")


if __name__ == "__main__":
    run_all_tests()
