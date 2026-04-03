"""Initial condition generators for the gravity simulation.

Central star is always particle index 0. Disk and cloud generators add
particles around the star with configurable angular momentum.
"""

from __future__ import annotations

import numpy as np

from .state import ParticleState

# Default G for circular orbit speed v = sqrt(G*M_star/r)
G_DEFAULT = 1.0


def _hernquist_enclosed(r: np.ndarray, M_halo: float, a_halo: float) -> np.ndarray:
    """Hernquist enclosed mass: M(<r) = M_halo * r^2 / (r + a)^2."""
    return M_halo * r ** 2 / (r + a_halo) ** 2


def make_disk_2d(
    n_particles: int,
    seed: int | None = None,
    M_star: float = 1.0,
    m_particle: float | None = None,
    r_min: float = 0.5,
    r_max: float = 2.0,
    G: float = G_DEFAULT,
    velocity_noise: float = 0.02,
    position_noise: float = 0.01,
    M_halo: float = 0.0,
    a_halo: float = 5.0,
) -> ParticleState:
    """Return a 2D state with central star at origin and particles in an annular disk.

    Particle 0 is the star (mass M_star, at origin, zero velocity). Particles 1..n
    are placed in [r_min, r_max] with circular orbit velocities v = sqrt(G*M_star/r)
    (prograde), plus small random perturbations.

    Parameters
    ----------
    n_particles : int
        Number of disk particles (total state size is n_particles + 1).
    seed : int or None
        Random seed for positions and perturbations.
    M_star : float
        Mass of the central star (code units).
    m_particle : float or None
        Mass per disk particle (code units). If None, uses 1/(n_particles+1) so
        total mass = 1 (backward compatible).
    r_min, r_max : float
        Inner and outer radius of the annulus.
    G : float
        Gravitational constant in code units.
    velocity_noise : float
        Scale of random fractional perturbation to velocities.
    position_noise : float
        Scale of random perturbation added to positions (in same units as r).
    """
    rng = np.random.default_rng(seed)
    n_total = n_particles + 1

    # Star at origin
    star_pos = np.zeros((1, 2), dtype=float)
    star_vel = np.zeros((1, 2), dtype=float)
    star_mass = np.array([M_star], dtype=float)

    # Uniform surface density: sample r^2 uniformly in [r_min^2, r_max^2]
    r2_min, r2_max = r_min * r_min, r_max * r_max
    r2 = r2_min + (r2_max - r2_min) * rng.random(n_particles, dtype=float)
    r = np.sqrt(r2)
    theta = 2.0 * np.pi * rng.random(n_particles, dtype=float)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.column_stack([x, y]).astype(float)
    positions += position_noise * rng.normal(size=(n_particles, 2))

    # Circular orbit speed accounting for star + halo enclosed mass
    M_enc = M_star + _hernquist_enclosed(r, M_halo, a_halo)
    v_circ = np.sqrt(G * M_enc / r)
    vx = -v_circ * np.sin(theta)
    vy = v_circ * np.cos(theta)
    velocities = np.column_stack([vx, vy]).astype(float)
    velocities += velocity_noise * v_circ[:, None] * rng.normal(size=(n_particles, 2))

    if m_particle is None:
        m_particle = 1.0 / n_total
    masses_disk = np.full(n_particles, m_particle, dtype=float)

    positions = np.vstack([star_pos, positions])
    velocities = np.vstack([star_vel, velocities])
    masses = np.concatenate([star_mass, masses_disk])

    return ParticleState(positions=positions, velocities=velocities, masses=masses)


def make_cloud_2d(
    n_particles: int,
    seed: int | None = None,
    M_star: float = 1.0,
    m_particle: float | None = None,
    r_max: float = 2.0,
    angular_fraction: float = 0.5,
    G: float = G_DEFAULT,
    M_halo: float = 0.0,
    a_halo: float = 5.0,
) -> ParticleState:
    """Return a 2D state with central star and a random particle cloud.

    Particle 0 is the star. Particles 1..n are placed randomly inside radius r_max
    (uniform in area: r from sqrt(uniform)*r_max, theta uniform). Each particle
    gets a tangential velocity equal to angular_fraction * v_circ(r), giving
    a less ordered configuration than the disk.

    Parameters
    ----------
    n_particles : int
        Number of cloud particles (total state size is n_particles + 1).
    seed : int or None
        Random seed.
    M_star : float
        Mass of the central star (code units).
    m_particle : float or None
        Mass per cloud particle (code units). If None, uses 1/(n_particles+1).
    r_max : float
        Maximum radius for cloud particles.
    angular_fraction : float
        Fraction of circular velocity applied tangentially (0 = radial infall, 1 = circular).
    G : float
        Gravitational constant in code units.
    """
    rng = np.random.default_rng(seed)
    n_total = n_particles + 1

    star_pos = np.zeros((1, 2), dtype=float)
    star_vel = np.zeros((1, 2), dtype=float)
    star_mass = np.array([M_star], dtype=float)

    # Uniform in area: r = sqrt(u) * r_max, u uniform in [0,1]
    r = np.sqrt(rng.random(n_particles, dtype=float)) * r_max
    theta = 2.0 * np.pi * rng.random(n_particles, dtype=float)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.column_stack([x, y]).astype(float)

    r_safe = np.maximum(r, 1e-8)
    M_enc = M_star + _hernquist_enclosed(r_safe, M_halo, a_halo)
    v_circ = np.sqrt(G * M_enc / r_safe)
    v_tangent = angular_fraction * v_circ
    vx = -v_tangent * np.sin(theta)
    vy = v_tangent * np.cos(theta)
    velocities = np.column_stack([vx, vy]).astype(float)

    if m_particle is None:
        m_particle = 1.0 / n_total
    masses_disk = np.full(n_particles, m_particle, dtype=float)

    positions = np.vstack([star_pos, positions])
    velocities = np.vstack([star_vel, velocities])
    masses = np.concatenate([star_mass, masses_disk])

    return ParticleState(positions=positions, velocities=velocities, masses=masses)


def make_disk_3d(
    n_particles: int,
    seed: int | None = None,
    M_star: float = 1.0,
    m_particle: float | None = None,
    r_min: float = 0.5,
    r_max: float = 2.0,
    thickness: float = 0.05,
    G: float = G_DEFAULT,
    velocity_noise: float = 0.02,
    position_noise: float = 0.01,
    M_halo: float = 0.0,
    a_halo: float = 5.0,
) -> ParticleState:
    """Return a 3D state: central star at origin and particles in a thickened disk.

    Same as make_disk_2d but with small z-extent (thickness) and vz perturbations.
    Particle 0 is the star. Positions and velocities have shape (N, 3).
    m_particle: mass per disk particle (code units); if None, uses 1/(n_particles+1).
    """
    rng = np.random.default_rng(seed)
    n_total = n_particles + 1

    star_pos = np.zeros((1, 3), dtype=float)
    star_vel = np.zeros((1, 3), dtype=float)
    star_mass = np.array([M_star], dtype=float)

    r2_min, r2_max = r_min * r_min, r_max * r_max
    r2 = r2_min + (r2_max - r2_min) * rng.random(n_particles, dtype=float)
    r = np.sqrt(r2)
    theta = 2.0 * np.pi * rng.random(n_particles, dtype=float)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = thickness * (rng.random(n_particles, dtype=float) - 0.5)
    positions = np.column_stack([x, y, z]).astype(float)
    positions += position_noise * rng.normal(size=(n_particles, 3))

    r_safe = np.maximum(r, 1e-8)
    M_enc = M_star + _hernquist_enclosed(r_safe, M_halo, a_halo)
    v_circ = np.sqrt(G * M_enc / r_safe)
    vx = -v_circ * np.sin(theta)
    vy = v_circ * np.cos(theta)
    vz = velocity_noise * v_circ * rng.normal(size=n_particles)
    velocities = np.column_stack([vx, vy, vz]).astype(float)
    velocities += velocity_noise * v_circ[:, None] * rng.normal(size=(n_particles, 3))

    if m_particle is None:
        m_particle = 1.0 / n_total
    masses_disk = np.full(n_particles, m_particle, dtype=float)

    positions = np.vstack([star_pos, positions])
    velocities = np.vstack([star_vel, velocities])
    masses = np.concatenate([star_mass, masses_disk])

    return ParticleState(positions=positions, velocities=velocities, masses=masses)


def make_cloud_3d(
    n_particles: int,
    seed: int | None = None,
    M_star: float = 1.0,
    m_particle: float | None = None,
    r_max: float = 2.0,
    angular_fraction: float = 0.5,
    G: float = G_DEFAULT,
    M_halo: float = 0.0,
    a_halo: float = 5.0,
) -> ParticleState:
    """Return a 3D state: central star and random particle cloud (uniform in volume).

    Particles 1..n are placed uniformly in a ball of radius r_max, with tangential
    velocity angular_fraction * v_circ. Positions and velocities have shape (N, 3).
    m_particle: mass per particle (code units); if None, uses 1/(n_particles+1).
    """
    rng = np.random.default_rng(seed)
    n_total = n_particles + 1

    star_pos = np.zeros((1, 3), dtype=float)
    star_vel = np.zeros((1, 3), dtype=float)
    star_mass = np.array([M_star], dtype=float)

    # Uniform in volume: r^3 uniform, then uniform on sphere
    u = rng.random(n_particles, dtype=float)
    r = (u ** (1.0 / 3.0)) * r_max
    phi = np.arccos(2.0 * rng.random(n_particles, dtype=float) - 1.0)
    theta = 2.0 * np.pi * rng.random(n_particles, dtype=float)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    positions = np.column_stack([x, y, z]).astype(float)

    r_safe = np.maximum(r, 1e-8)
    M_enc = M_star + _hernquist_enclosed(r_safe, M_halo, a_halo)
    v_circ = np.sqrt(G * M_enc / r_safe)
    v_tangent = angular_fraction * v_circ
    vx = -v_tangent * np.sin(theta)
    vy = v_tangent * np.cos(theta)
    vz = np.zeros(n_particles, dtype=float)
    velocities = np.column_stack([vx, vy, vz]).astype(float)

    if m_particle is None:
        m_particle = 1.0 / n_total
    masses_disk = np.full(n_particles, m_particle, dtype=float)

    positions = np.vstack([star_pos, positions])
    velocities = np.vstack([star_vel, velocities])
    masses = np.concatenate([star_mass, masses_disk])

    return ParticleState(positions=positions, velocities=velocities, masses=masses)


def make_uniform_2d(n: int, seed: int | None = None) -> ParticleState:
    """Return a nearly uniform 2D particle distribution with tiny noise.

    This is our first "early-universe-like" initial condition:
    - particles are roughly uniformly distributed in a unit square
    - small Gaussian noise adds tiny perturbations
    """
    rng = np.random.default_rng(seed)

    # Base uniform distribution in [0, 1) x [0, 1)
    positions = rng.random((n, 2), dtype=float)

    # Add small perturbations and wrap back into the box
    positions += 0.02 * rng.normal(size=(n, 2))
    positions %= 1.0

    velocities = np.zeros((n, 2), dtype=float)
    masses = np.ones(n, dtype=float) / n

    return ParticleState(positions=positions, velocities=velocities, masses=masses)

