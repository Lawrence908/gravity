"""Time integration methods for the gravity simulation (Phase 1: 2D)."""

from __future__ import annotations

from .state import AccelerationFn, ParticleState


def euler_step(
    state: ParticleState,
    dt: float,
    accel_fn: AccelerationFn,
) -> ParticleState:
    """Very simple Euler integrator (for early testing only)."""
    acc = accel_fn(state)
    new_vel = state.velocities + dt * acc
    new_pos = state.positions + dt * new_vel
    return ParticleState(positions=new_pos, velocities=new_vel, masses=state.masses)


def leapfrog_step(
    state: ParticleState,
    dt: float,
    accel_fn: AccelerationFn,
) -> ParticleState:
    """Leapfrog (kick-drift-kick) integrator.

    More stable than Euler for gravitational dynamics and good for testing
    energy conservation.
    """
    # First half-kick
    a0 = accel_fn(state)
    v_half = state.velocities + 0.5 * dt * a0

    # Drift
    pos_new = state.positions + dt * v_half
    mid_state = ParticleState(positions=pos_new, velocities=v_half, masses=state.masses)

    # Second half-kick
    a1 = accel_fn(mid_state)
    v_new = v_half + 0.5 * dt * a1

    return ParticleState(positions=pos_new, velocities=v_new, masses=state.masses)
