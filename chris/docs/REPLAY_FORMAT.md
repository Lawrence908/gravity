# Replay File Format

Replay files store simulation snapshots (particle positions at fixed step intervals) plus metadata so that a viewer can replay the run without re-running the simulation.

## Format: NumPy .npz

The replay is a single `.npz` file (e.g. `outputs/runs/run.npz`) produced by `gravity.replay.save_replay` and loaded with `gravity.replay.load_replay`.

## Constant N (no collisions)

When the run has no mergers, every snapshot has the same particle count N.

| Key           | Shape / type   | Description |
|---------------|----------------|-------------|
| `positions`   | (n_snapshots, N, 2) or (n_snapshots, N, 3) | Position of each particle at each saved step; index 0 is the central star in disk/cloud runs. |
| `steps`       | (n_snapshots,) int64 | Step index for each snapshot. |
| `masses`      | (N,) float64   | Particle masses (constant for the run). |
| `dt`          | scalar float64 | Timestep. |
| `softening`   | scalar float64 | Gravitational softening length. |
| `G`           | scalar float64 | Gravitational constant in code units. |
| `n_particles` | scalar int64   | N. |
| `n_snapshots` | scalar int64   | Number of saved frames. |

## Variable N (collisions enabled)

When the run uses `--collisions`, mergers reduce N over time. The file stores concatenated arrays and per-snapshot counts.

| Key                       | Shape / type   | Description |
|---------------------------|----------------|-------------|
| `positions`               | (sum of N_t, 2) or (sum of N_t, 3) | All positions concatenated; snapshot k is `positions[snapshot_offset[k]:snapshot_offset[k+1]]`. |
| `masses`                  | (sum of N_t,) float64 | All masses concatenated; same slicing by `snapshot_offset`. |
| `n_particles_per_snapshot`| (n_snapshots,) int64 | Particle count for each snapshot. |
| `snapshot_offset`         | (n_snapshots+1,) int64 | Start index for each snapshot in `positions`/`masses` (cumulative). |
| `steps`                   | (n_snapshots,) int64 | Step index for each snapshot. |
| `dt`, `softening`, `G`    | scalars       | As above. |
| `n_snapshots`             | scalar int64  | Number of saved frames. |

`load_replay` returns `variable_n: True` and `positions`/`masses` as **lists of arrays** (one per snapshot). Index 0 of each snapshot is the star.

## Backward compatibility

- **Loading:** If the file has no `n_particles_per_snapshot` key, `load_replay` uses the constant-N layout and returns `positions` as a 3D array, `masses` as 1D, and `variable_n: False`.
- **Saving:** Without `masses_per_snapshot`, `save_replay` writes the constant-N format.

## Large replays and browser loading

Replay JSON is loaded fully into the browser. Very large files (e.g. 5000 particles × 50k steps with replay_every=20 → 2500+ snapshots, ~700 MB) can fail to load or exhaust memory. To avoid this:

- **When running:** Use a larger "Snapshot every N steps" for long runs (e.g. 50–100 for 50k steps so snapshots stay under ~2000). The sim server caps at 2000 snapshots so runs from the web UI stay loadable.
- **After the fact:** Thin an existing replay with `python tools/thin_replay.py replay.npz out.json --every 50` or `--max-snapshots 2000`, or re-export with `python tools/export_replay_to_json.py replay.npz out.json --every 50`.

## Usage

- **Export:** `python -m gravity.demo_2d --save-replay outputs/runs/run.npz --replay-every 20 --no-viz --steps 1000`
- **With collisions:** add `--collisions` (and optionally `--r-collide`); replay will be variable-N.
- **Load in Python:** `from gravity.replay import load_replay; data = load_replay("outputs/runs/run.npz")` then `data["positions"]`, `data["steps"]`, etc. Use `data["variable_n"]` to branch on constant vs variable N.
- **Web viewer:** The WebGL viewer accepts both formats; export script and viewer handle variable N per frame.

## Notes

- Velocities are not stored; the format is position-only for simplicity and smaller files.
- For 2D runs, position vectors have length 2; for 3D, length 3.
