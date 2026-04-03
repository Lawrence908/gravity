"""Replay file format and I/O for the gravity simulation.

See docs/REPLAY_FORMAT.md for the .npz layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .state import ParticleState


def save_replay(
    path: str | Path,
    positions_list: list[np.ndarray],
    step_indices: list[int],
    masses: np.ndarray,
    dt: float,
    softening: float = 0.05,
    G: float = 1.0,
    masses_per_snapshot: list[np.ndarray] | None = None,
) -> None:
    """Write a replay .npz file.

    Parameters
    ----------
    path : path to output .npz file
    positions_list : list of position arrays, each shape (N_t, 2) or (N_t, 3)
    step_indices : step number for each snapshot (length = len(positions_list))
    masses : array of shape (N,) for last snapshot (or when constant N)
    dt, softening, G : simulation parameters
    masses_per_snapshot : if provided, each snapshot can have different N (e.g. after collisions).
        Length must equal len(positions_list). Enables variable-N format in .npz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = np.array(step_indices, dtype=np.int64)
    n_snapshots = len(positions_list)

    if masses_per_snapshot is not None and len(masses_per_snapshot) == n_snapshots:
        # Variable N: concatenate positions and masses; store per-snapshot counts and offset
        n_per = np.array([p.shape[0] for p in positions_list], dtype=np.int64)
        snapshot_offset = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)
        positions_cat = np.concatenate(positions_list, axis=0)
        masses_cat = np.concatenate(masses_per_snapshot, axis=0)
        np.savez(
            path,
            positions=positions_cat,
            masses=masses_cat,
            n_particles_per_snapshot=n_per,
            snapshot_offset=snapshot_offset,
            steps=steps,
            dt=np.float64(dt),
            softening=np.float64(softening),
            G=np.float64(G),
            n_snapshots=np.int64(n_snapshots),
        )
    else:
        # Constant N (original format)
        positions = np.stack(positions_list, axis=0)
        np.savez(
            path,
            positions=positions,
            steps=steps,
            masses=masses,
            dt=np.float64(dt),
            softening=np.float64(softening),
            G=np.float64(G),
            n_particles=np.int64(masses.shape[0]),
            n_snapshots=np.int64(n_snapshots),
        )


def load_replay(path: str | Path) -> dict:
    """Load a replay .npz file. Returns a dict with keys:

    - positions: (n_snapshots, N, 2) or (n_snapshots, N, 3) for constant N; or list of
      arrays (variable N) when file was saved with masses_per_snapshot
    - steps: (n_snapshots,)
    - masses: (N,) for constant N, or list of (N_t,) for variable N
    - dt, softening, G: scalars
    - n_particles: int (constant N only)
    - n_snapshots: int
    - variable_n: bool, True if per-snapshot particle count varies
    """
    data = np.load(path, allow_pickle=False)
    n_snapshots = int(data["n_snapshots"])

    if "n_particles_per_snapshot" in data:
        n_per = data["n_particles_per_snapshot"]
        offset = data["snapshot_offset"]
        pos_cat = data["positions"]
        mass_cat = data["masses"]
        positions_list = [
            pos_cat[offset[i] : offset[i + 1]].copy() for i in range(n_snapshots)
        ]
        masses_list = [
            mass_cat[offset[i] : offset[i + 1]].copy() for i in range(n_snapshots)
        ]
        return {
            "positions": positions_list,
            "masses": masses_list,
            "steps": data["steps"],
            "dt": float(data["dt"]),
            "softening": float(data["softening"]),
            "G": float(data["G"]),
            "n_snapshots": n_snapshots,
            "variable_n": True,
        }
    else:
        return {
            "positions": data["positions"],
            "steps": data["steps"],
            "masses": data["masses"],
            "dt": float(data["dt"]),
            "softening": float(data["softening"]),
            "G": float(data["G"]),
            "n_particles": int(data["n_particles"]),
            "n_snapshots": n_snapshots,
            "variable_n": False,
        }
