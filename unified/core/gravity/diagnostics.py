"""Diagnostics for the gravity simulation: energy, angular momentum, and logging."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import ParticleState


def compute_kinetic_energy(state: ParticleState) -> float:
    """Return total kinetic energy K = (1/2) sum_i m_i v_i^2."""
    v2 = np.sum(state.velocities * state.velocities, axis=1)
    return 0.5 * float(np.sum(state.masses * v2))


def compute_potential_energy(
    state: ParticleState,
    softening: float = 0.05,
    G: float = 1.0,
) -> float:
    """Return total gravitational potential energy (softened, pair-summed)."""
    pos = state.positions
    masses = state.masses
    n = pos.shape[0]

    U = 0.0
    for i in range(n):
        dx = pos[i + 1 :] - pos[i]
        if dx.size == 0:
            continue
        r2 = np.sum(dx * dx, axis=1) + softening * softening
        r = np.sqrt(r2)
        U -= float(np.sum(G * masses[i] * masses[i + 1 :] / r))

    return U


def compute_total_energy(
    state: ParticleState,
    softening: float = 0.05,
    G: float = 1.0,
) -> float:
    """Return total (kinetic + potential) energy."""
    return compute_kinetic_energy(state) + compute_potential_energy(
        state, softening=softening, G=G
    )


def compute_angular_momentum(state: ParticleState) -> float:
    """Return total z-component of angular momentum (L_z = sum m_i (x_i v_yi - y_i v_xi))."""
    pos = state.positions
    vel = state.velocities
    Lz_per = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    return float(np.sum(state.masses * Lz_per))


def compute_angular_momentum_vector(state: ParticleState) -> np.ndarray:
    """Return total angular momentum vector L = sum_i m_i (r_i x v_i). For 3D only."""
    pos = state.positions
    vel = state.velocities
    if pos.shape[1] != 3:
        raise ValueError("compute_angular_momentum_vector requires 3D state (positions shape (N, 3))")
    # L = r x v per particle: Lx = y*vz - z*vy, Ly = z*vx - x*vz, Lz = x*vy - y*vx
    Lx = pos[:, 1] * vel[:, 2] - pos[:, 2] * vel[:, 1]
    Ly = pos[:, 2] * vel[:, 0] - pos[:, 0] * vel[:, 2]
    Lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    return np.array([
        float(np.sum(state.masses * Lx)),
        float(np.sum(state.masses * Ly)),
        float(np.sum(state.masses * Lz)),
    ])


@dataclass
class SimulationLog:
    """Accumulate time-series of step, energy, and angular momentum for a run."""

    steps: list[int] = field(default_factory=list)
    kinetic: list[float] = field(default_factory=list)
    potential: list[float] = field(default_factory=list)
    total_energy: list[float] = field(default_factory=list)
    angular_momentum: list[float] = field(default_factory=list)

    def append(
        self,
        step: int,
        state: ParticleState,
        softening: float = 0.05,
        G: float = 1.0,
    ) -> None:
        """Record diagnostics for the current step."""
        self.steps.append(step)
        K = compute_kinetic_energy(state)
        U = compute_potential_energy(state, softening=softening, G=G)
        self.kinetic.append(K)
        self.potential.append(U)
        self.total_energy.append(K + U)
        self.angular_momentum.append(compute_angular_momentum(state))

    def summary_plot(self, path: str | None = None) -> None:
        """Plot E and L vs step; save to path if given. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(self.steps, self.total_energy, label="E total")
        ax1.plot(self.steps, self.kinetic, alpha=0.7, label="K")
        ax1.plot(self.steps, self.potential, alpha=0.7, label="U")
        ax1.set_ylabel("Energy")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax2.plot(self.steps, self.angular_momentum, color="green")
        ax2.set_ylabel("Angular momentum (Lz)")
        ax2.set_xlabel("Step")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close(fig)
