"""Simple 3D visualization for the gravity simulation (mplot3d)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .state import ParticleState


class LiveScatter3D:
    """Live 3D scatter: central star (index 0) as distinct marker, particles as dots."""

    def __init__(self, r_max: float = 2.0) -> None:
        self.r_max = r_max
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.scat_star = None
        self.scat_particles = None
        margin = 0.2 * r_max
        self.ax.set_xlim(-r_max - margin, r_max + margin)
        self.ax.set_ylim(-r_max - margin, r_max + margin)
        self.ax.set_zlim(-r_max - margin, r_max + margin)
        self.ax.set_title("3D gravity — step 0")

    def update(
        self,
        state: ParticleState,
        step: int = 0,
        E: float | None = None,
        L: float | None = None,
    ) -> None:
        pos = state.positions
        n = pos.shape[0]
        if n == 0:
            return

        star_pos = pos[0:1]
        part_pos = pos[1:]

        if self.scat_star is None:
            self.scat_star = self.ax.scatter(
                star_pos[:, 0],
                star_pos[:, 1],
                star_pos[:, 2],
                s=120,
                c="yellow",
                marker="*",
                edgecolors="orange",
                label="star",
            )
        else:
            self.scat_star.set_offsets(star_pos[:, :2])
            self.scat_star.set_3d_properties(star_pos[:, 2], "z")

        if part_pos.shape[0] > 0:
            if self.scat_particles is None:
                self.scat_particles = self.ax.scatter(
                    part_pos[:, 0],
                    part_pos[:, 1],
                    part_pos[:, 2],
                    s=2,
                    c="blue",
                    alpha=0.7,
                )
            else:
                self.scat_particles.set_offsets(part_pos[:, :2])
                self.scat_particles.set_3d_properties(part_pos[:, 2], "z")

        title = f"3D gravity — step {step}"
        if E is not None:
            title += f"  E={E:.4g}"
        if L is not None:
            title += f"  Lz={L:.4g}"
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self) -> None:
        plt.show()
