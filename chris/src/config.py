"""Configuration and shared constants for the gravity simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GravityConfig:
    """Configuration for the gravity simulation (Phase 1: 2D; Phase 2: 3D).

    Recommended presets for spiral-galaxy-like runs (1k-5k particles):
        M_star=1.0, m_particle ~ M_star/1000 to M_star/10000,
        M_halo=5-20 (dominant mass for flat rotation curve),
        a_halo ~ r_max * 0.5,
        r_min=2.0, r_max=15.0, softening=0.05-0.1,
        dt=0.01-0.02, velocity_noise=0.05.
    """

    dim: int = 2  # 2D for Phase 1, extend to 3D later
    n_particles: int = 1000
    time_step: float = 0.01
    softening_length: float = 0.05
    M_star: float = 1.0
    r_min: float = 0.5
    r_max: float = 2.0
    ic_type: str = "disk"  # "disk" or "cloud"
    M_halo: float = 0.0  # dark-matter halo mass (Hernquist profile, 0 = off)
    a_halo: float = 5.0  # halo scale radius


@dataclass
class AppConfig:
    """Top-level configuration wrapper."""

    gravity: GravityConfig = GravityConfig()
