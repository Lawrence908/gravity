"""Live visualization for the 2D gravity demo: central star + particles."""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from .state import ParticleState


class LiveScatter2D:
    """Live 2D scatter: central star (index 0) as distinct marker, particles colored by distance or speed."""

    def __init__(
        self,
        r_max: float = 2.0,
        color_by: str = "distance",
    ) -> None:
        self.r_max = r_max
        self.color_by = color_by
        self.fig, self.ax = plt.subplots()
        self.scat_particles = None
        self.scat_star = None
        self.ax.set_aspect("equal", adjustable="box")
        margin = 0.2 * self.r_max
        self.ax.set_xlim(-self.r_max - margin, self.r_max + margin)
        self.ax.set_ylim(-self.r_max - margin, self.r_max + margin)
        self.ax.set_title("2D gravity — step 0")

    def update(
        self,
        state: ParticleState,
        step: int = 0,
        E: float | None = None,
        L: float | None = None,
    ) -> None:
        pos = state.positions
        vel = state.velocities
        n = pos.shape[0]

        if n == 0:
            return

        # Central star is index 0
        star_pos = pos[0:1]
        part_pos = pos[1:]
        n_part = part_pos.shape[0]

        if self.scat_star is None:
            self.scat_star = self.ax.scatter(
                star_pos[:, 0],
                star_pos[:, 1],
                s=120,
                c="yellow",
                marker="*",
                edgecolors="orange",
                zorder=5,
                label="star",
            )
        else:
            self.scat_star.set_offsets(star_pos)

        if n_part > 0:
            if self.color_by == "distance":
                r = np.sqrt(np.sum(part_pos * part_pos, axis=1))
                c = r
            else:
                v2 = np.sum(vel[1:] * vel[1:], axis=1)
                c = np.sqrt(v2)
            if self.scat_particles is None:
                self.scat_particles = self.ax.scatter(
                    part_pos[:, 0],
                    part_pos[:, 1],
                    s=2,
                    c=c,
                    cmap="viridis",
                    alpha=0.8,
                )
            else:
                self.scat_particles.set_offsets(part_pos)
                self.scat_particles.set_array(c)

        title = f"2D gravity — step {step}"
        if E is not None:
            title += f"  E={E:.4g}"
        if L is not None:
            title += f"  Lz={L:.4g}"
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show(self) -> None:
        plt.show()
