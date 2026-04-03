"""Frame export for the 2D gravity demo: save PNG snapshots to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .state import ParticleState


def save_frame(
    state: ParticleState,
    out_path: Path,
    step: int,
    r_max: float = 2.0,
    E: float | None = None,
    L: float | None = None,
    color_by: str = "distance",
) -> None:
    """Render current state to a PNG file (star + particles, same style as LiveScatter2D).

    Parameters
    ----------
    state : ParticleState
        Current positions, velocities, masses.
    out_path : Path
        Output file path (e.g. outputs/frames/frame_00042.png).
    step : int
        Step number for title.
    r_max : float
        Axis limit scale (plot shows ±(r_max + margin)).
    E, L : float or None
        Optional energy and angular momentum for title.
    color_by : str
        "distance" or "speed" for particle coloring.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos = state.positions
    vel = state.velocities
    n = pos.shape[0]
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_xlim(-r_max - 0.5, r_max + 0.5)
        ax.set_ylim(-r_max - 0.5, r_max + 0.5)
        ax.set_title(f"2D gravity — step {step}")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return

    star_pos = pos[0:1]
    part_pos = pos[1:]
    n_part = part_pos.shape[0]

    fig, ax = plt.subplots()
    margin = 0.2 * r_max
    ax.set_xlim(-r_max - margin, r_max + margin)
    ax.set_ylim(-r_max - margin, r_max + margin)
    ax.set_aspect("equal", adjustable="box")

    ax.scatter(
        star_pos[:, 0],
        star_pos[:, 1],
        s=120,
        c="yellow",
        marker="*",
        edgecolors="orange",
        zorder=5,
        label="star",
    )

    if n_part > 0:
        if color_by == "distance":
            r = np.sqrt(np.sum(part_pos * part_pos, axis=1))
            c = r
        else:
            v2 = np.sum(vel[1:] * vel[1:], axis=1)
            c = np.sqrt(v2)
        ax.scatter(
            part_pos[:, 0],
            part_pos[:, 1],
            s=2,
            c=c,
            cmap="viridis",
            alpha=0.8,
        )

    title = f"2D gravity — step {step}"
    if E is not None:
        title += f"  E={E:.4g}"
    if L is not None:
        title += f"  Lz={L:.4g}"
    ax.set_title(title)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
