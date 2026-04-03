"""Long-run stability test for Phase 1: disk run with bounded motion and conservation checks.

Run with:

    cd src
    python -m gravity.test_long_run

Serves as a documented stability certificate for the 2D CPU model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .diagnostics import (
    SimulationLog,
    compute_angular_momentum,
    compute_total_energy,
)
from .forces_cpu import compute_accelerations_vectorized
from .init_conditions import make_disk_2d
from .integrators import leapfrog_step
from .state import ParticleState


# Defaults aligned with demo_2d
R_MAX = 2.0
N_PARTICLES = 500
N_STEPS = 2000  # enough to verify stability; use 5000+ for a full certificate run
DT = 0.01
SOFTENING = 0.05
LOG_INTERVAL = 100
# Bounded motion: no particle may exceed this radius (catches blow-ups, not tight confinement).
# Disk can spread over long runs; use a large limit so we only fail on true instability.
ESCAPE_RADIUS_MULTIPLIER = 50.0
MAX_ESCAPE_RADIUS = ESCAPE_RADIUS_MULTIPLIER * R_MAX
# Relative drift thresholds (intentionally loose for long run)
MAX_RELATIVE_E_DRIFT = 0.05
MAX_RELATIVE_L_DRIFT = 0.05


def run_long_run(
    n_particles: int = N_PARTICLES,
    n_steps: int = N_STEPS,
    dt: float = DT,
    r_max: float = R_MAX,
    softening: float = SOFTENING,
    seed: int = 123,
    log_interval: int = LOG_INTERVAL,
    max_escape_radius: float | None = None,
    max_rel_E_drift: float = MAX_RELATIVE_E_DRIFT,
    max_rel_L_drift: float = MAX_RELATIVE_L_DRIFT,
    save_plot_path: str | None = "outputs/long_run_diagnostics.png",
) -> None:
    """Run a long disk simulation and verify stability and conservation.

    Raises
    ------
    AssertionError
        If any particle escapes beyond max_escape_radius or if E/L drift exceeds thresholds.
    """
    if max_escape_radius is None:
        max_escape_radius = ESCAPE_RADIUS_MULTIPLIER * r_max

    state = make_disk_2d(
        n_particles,
        seed=seed,
        M_star=1.0,
        r_min=0.5,
        r_max=r_max,
    )

    def accel_fn(s: ParticleState) -> np.ndarray:
        return compute_accelerations_vectorized(
            s, softening=softening, G=1.0
        )

    sim_log = SimulationLog()
    E0 = compute_total_energy(state, softening=softening, G=1.0)
    L0 = compute_angular_momentum(state)

    for step in range(n_steps):
        state = leapfrog_step(state, dt=dt, accel_fn=accel_fn)

        if step % log_interval == 0:
            sim_log.append(step, state, softening=softening, G=1.0)

        # Bounded motion: particles (index 1 onward) must stay within max_escape_radius
        part_pos = state.positions[1:]
        r = np.sqrt(np.sum(part_pos * part_pos, axis=1))
        max_r = float(np.max(r))
        assert max_r <= max_escape_radius, (
            f"Particle escaped: max radius {max_r:.4g} > {max_escape_radius:.4g} at step {step}"
        )

    sim_log.append(n_steps, state, softening=softening, G=1.0)

    part_pos_final = state.positions[1:]
    max_r = float(np.max(np.sqrt(np.sum(part_pos_final * part_pos_final, axis=1))))

    E_final = compute_total_energy(state, softening=softening, G=1.0)
    L_final = compute_angular_momentum(state)
    rel_E_drift = abs(E_final - E0) / max(1.0, abs(E0))
    rel_L_drift = abs(L_final - L0) / max(1e-20, abs(L0))

    assert rel_E_drift <= max_rel_E_drift, (
        f"Energy drift {rel_E_drift:.4g} exceeds limit {max_rel_E_drift:.4g}"
    )
    assert rel_L_drift <= max_rel_L_drift, (
        f"Angular momentum drift {rel_L_drift:.4g} exceeds limit {max_rel_L_drift:.4g}"
    )

    print(f"Long-run stability test PASSED ({n_steps} steps, n={n_particles})")
    print(f"  E0={E0:.6g}, E_final={E_final:.6g}, rel_E_drift={rel_E_drift:.3e}")
    print(f"  L0={L0:.6g}, L_final={L_final:.6g}, rel_L_drift={rel_L_drift:.3e}")
    print(f"  Max particle radius = {max_r:.4g} (limit {max_escape_radius:.4g})")

    if save_plot_path:
        path = Path(save_plot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sim_log.summary_plot(path=str(path))
        print(f"  Conservation plot saved to {path}")


def main() -> None:
    run_long_run()


if __name__ == "__main__":
    main()
