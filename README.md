# Unified Gravity Simulation Platform

ASTR 311 group project — unified frontend and backend for N-body gravitational
simulations by Chris, Ethan, Jasper, and Brad.

## Quick Start

```bash
cd unified

# Install dependencies
pip install -r backend/requirements.txt

# Run the server
uvicorn backend.controller:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

## Structure

```
unified/
  frontend/           Web UI (served at /)
    index.html        Dashboard — simulation picker and replay library
    viewer.html       Merged viewer (replay playback + real-time WebSocket)
    replays/          Pre-computed .json replay files
  backend/
    controller.py     FastAPI server: REST + WebSocket
    requirements.txt
  core/
    gravity/          Shared N-body engine (NumPy, leapfrog)
    common/           Shared utilities
    config.py         GravityConfig dataclass
  simulations/
    chris/app.py      CPU N-body (disk/cloud galaxy scenarios)
    ethan/app.py      3-body problem + Pluto system
    jasper/app.py     GPU Yoshida symplectic + GR corrections
    brad/app.py       Binary star template
```

## Adding a Simulation

1. Create `simulations/yourname/app.py`
2. Implement an `AsyncSimulator` class with these methods:
   - `get_latest_state() -> dict | None`
   - `send_command(cmd: dict) -> None`
   - `calculate_velocity(position, mass) -> list`
   - `stop() -> None`
3. Register it in `backend/controller.py` `SIMULATION_REGISTRY`

The state dict returned by `get_latest_state()` should have this shape:

```python
{
    "bodies": {
        "positions": [[x, y, z], ...],  # or [[x, y], ...] for 2D
        "velocities": [[vx, vy, vz], ...],
        "masses": [m1, m2, ...],
        "radii": [r1, r2, ...],
        "colors": [[h, s, l], ...],     # HSL values
        "names": ["Sun", "Earth", ...],
    },
    "params": {"G": ..., "dt": ..., ...},
    "performance": {"compute_time_ms": ..., "avg_fps": ..., "body_count": ...},
}
```

## Modes

**Replay mode**: Load pre-computed `.json` files from `frontend/replays/`.
Timeline scrubbing, playback speed control, trails.

**Real-time mode**: WebSocket connection to a running simulation.
Live rendering at physics loop speed, add bodies, reset, adjust parameters.

## Replay File Format

JSON with:
- `positions`: array of snapshots, each `[[x,y,z], ...]` (particle 0 = star)
- `steps`: step indices per snapshot
- `masses`: particle masses
- `dt`, `n_snapshots`, `n_particles`, `variable_n`
