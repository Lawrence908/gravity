"""Shared N-body gravity simulation engine."""

from .state import ParticleState, AccelerationFn
from .forces_cpu import compute_accelerations, compute_accelerations_vectorized, compute_halo_acceleration
from .integrators import leapfrog_step
from .collisions import resolve_collisions
from .init_conditions import make_disk_2d, make_cloud_2d, make_disk_3d, make_cloud_3d
from .replay import save_replay, load_replay
from .diagnostics import compute_total_energy, compute_angular_momentum

__all__ = [
    "ParticleState",
    "AccelerationFn",
    "compute_accelerations",
    "compute_accelerations_vectorized",
    "compute_halo_acceleration",
    "leapfrog_step",
    "resolve_collisions",
    "make_disk_2d",
    "make_cloud_2d",
    "make_disk_3d",
    "make_cloud_3d",
    "save_replay",
    "load_replay",
    "compute_total_energy",
    "compute_angular_momentum",
]
