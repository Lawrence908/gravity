# Ethan Simulator — 3‑Body Problem (Unified Platform)

This document explains the **chaotic three‑body problem simulation** implemented in `unified/simulations/ethan/app.py`, and how it plugs into the unified web platform.

If you’ve never seen this repo before, start here:

- **Physics + integrator + scenarios** live in `unified/simulations/ethan/app.py`
- **Backend controller** that loads the simulator lives in `unified/backend/controller.py`
- **Frontend** that launches real‑time sims lives in `unified/frontend/` (notably `index.html` + `viewer.html`)

---

## What you are looking at

Ethan’s module provides **small‑N gravitational systems** in 2D:

- **`three_body`**: a classic chaotic 3‑body configuration (“Pythagorean 3‑body”)
- **`pluto_system`**: Pluto–Charon plus 4 moons (same engine, different units/parameters)

This README focuses on **`three_body`**.

---

## File map (where the 3‑body code is)

All in one file:

- `unified/simulations/ethan/app.py`
  - `Body`: per‑body state (mass, position, velocity, name)
  - `_gravitational_acceleration(...)`: Newtonian acceleration on one body from all others (pairwise sum)
  - `create_three_body(...)`: builds the Pythagorean 3‑body initial condition
  - `SCENARIOS["three_body"]`: parameter bundle for the 3‑body run (G, dt, softening, factory)
  - `SmallNSimulator`: the Velocity‑Verlet integrator + state serialization for the viewer
  - `AsyncSimulator`: thread wrapper that runs physics continuously and streams latest states

The unified platform integration points:

- `unified/backend/controller.py`
  - `SIMULATION_REGISTRY["ethansim"]` declares `"scenarios": ["three_body", ...]`
  - `SimulationRunner` imports `simulations.ethan.app` and instantiates `AsyncSimulator(scenario=...)`
  - `/ws` websocket streams `get_latest_state()` results and forwards commands to `send_command(...)`

---

## Physics model (3‑body)

### Force law

The simulation uses softened Newtonian gravity. For a body \(i\), the acceleration is the sum over all other bodies \(j \ne i\):

\[
\mathbf{a}_i = G \sum_{j \ne i} m_j \, \frac{\mathbf{r}_j - \mathbf{r}_i}{\left(|\mathbf{r}_j - \mathbf{r}_i|^2 + \varepsilon^2\right)^{3/2}}
\]

Where:

- \(G\) is the gravitational constant used by the scenario
- \(\varepsilon\) is the **softening length** (`softening`) to avoid singularities at very small separations

Implementation: `_gravitational_acceleration(body, bodies, G, softening=...)`.

### Initial condition: “Pythagorean 3‑body”

`create_three_body()` builds 3 masses placed at the vertices of a **3‑4‑5 right triangle** and (by default) starts them at rest:

- masses: `[3, 4, 5]`
- positions: `[[1, 3], [-2, -1], [1, -1]]`
- velocities: `[[0, 0], [0, 0], [0, 0]]`

It then shifts into the **center‑of‑mass frame** by subtracting CoM position and velocity so the overall system isn’t drifting.

This setup is famous because it quickly becomes chaotic (close encounters, ejections, temporary binaries).

### Units (important!)

For `three_body`, the scenario uses **dimensionless / normalized units**:

- `G = 1.0`
- distances, masses, time are unitless

This is intentional: it’s numerically stable and easy to visualize.

---

## Integrator: Velocity‑Verlet (symplectic-ish, good for orbits)

`SmallNSimulator.step()` advances the system using Velocity‑Verlet:

1. Drift positions with current velocities and accelerations:
   - \( \mathbf{x} \leftarrow \mathbf{x} + \mathbf{v}\,dt + \frac{1}{2}\mathbf{a}\,dt^2 \)
2. Recompute accelerations using new positions
3. Kick velocities using average of old/new accelerations:
   - \( \mathbf{v} \leftarrow \mathbf{v} + \frac{1}{2}(\mathbf{a}_{old} + \mathbf{a}_{new})\,dt \)

Why this matters:

- Much better long‑run orbital behavior than Euler
- Usually keeps energy drift under control for reasonable `dt`

---

## Scenario configuration (the knobs)

Scenarios are defined in the `SCENARIOS` dict in `unified/simulations/ethan/app.py`.

For **3‑body**, these are the defaults:

- `scenario id`: `"three_body"`
- `G`: `1.0`
- `dt`: `0.001`
- `softening`: `0.01`
- `factory`: `create_three_body`

### What to tweak first

- **`dt`**:
  - Smaller = more accurate, slower
  - Larger = faster, but can destabilize close encounters
- **`softening`**:
  - Helps with very close passes (prevents huge accelerations)
  - Too large will “blur out” the physics and reduce chaos
- **initial velocities**:
  - A tiny perturbation can drastically change the trajectory (chaos)

---

## Runtime architecture (how it runs in the unified server)

### `AsyncSimulator` (threaded physics loop)

The unified controller expects each simulation module to expose an `AsyncSimulator` with methods:

- `get_latest_state() -> dict | None`
- `send_command(cmd: dict) -> None`
- `calculate_velocity(position, mass) -> list` (used by UI features; Ethan’s currently returns `[0, 0]`)
- `stop() -> None`

Ethan’s `AsyncSimulator`:

- Creates a `SmallNSimulator`
- Starts a dedicated **physics thread** (`_physics_loop`)
- Uses two queues:
  - `command_queue`: UI → simulation commands (reset, add body, update params)
  - `state_queue` (maxsize 2): latest states; older ones may be dropped if the viewer can’t keep up

### Controller wiring

`unified/backend/controller.py`:

- Imports the module from `SIMULATION_REGISTRY["ethansim"]["module"] == "simulations.ethan.app"`
- Passes websocket query `scenario=...` into `AsyncSimulator(scenario=scenario)`
- Streams packed states to the frontend over `/ws`

---

## WebSocket state schema (what the viewer receives)

`SmallNSimulator.get_state()` returns a dict shaped like:

```json
{
  "bodies": {
    "positions": [[x, y], ...],
    "velocities": [[vx, vy], ...],
    "masses": [m1, m2, ...],
    "radii": [r1, r2, ...],
    "colors": [[h, s, l], ...],
    "names": ["Body 1", "Body 2", "Body 3", ...]
  },
  "params": {
    "G": 1.0,
    "dt": 0.001,
    "scenario": "three_body"
  },
  "performance": {
    "compute_time_ms": 0.123,
    "avg_fps": 120,
    "body_count": 3
  }
}
```

Notes:

- This is **2D**, so positions/velocities are length‑2 arrays.
- Colors are **HSL** triples used by the viewer.
- Radii are derived from mass (purely visual scaling).

---

## WebSocket commands (what the viewer can send)

The controller forwards arbitrary JSON dicts from the websocket to `AsyncSimulator.send_command(cmd)`.

Ethan’s `_handle_command` supports:

### Important: 2D simulator vs 3D viewer inputs

The unified viewer UI uses **(x, y, z)** inputs for “Add Body”. Ethan’s `three_body` simulation is **3D** (with z=0 by default), while `pluto_system` is still **2D**.

- For `three_body`, send 3D vectors: `pos: [x, y, z]` and `vel: [vx, vy, vz]`.
- For `pluto_system`, send 2D vectors: `pos: [x, y]` and `vel: [vx, vy]`.

The simulator will coerce vectors to the current scenario dimension (trim/pad), but it’s best to send the correct length.

### Update parameters

```json
{"type": "update_params", "data": {"dt": 0.0005}}
```

Supported fields:

- `dt` (currently the only one implemented)

### Add a body

```json
{
  "type": "add_body",
  "data": {
    "pos": [0.0, 1.0],
    "vel": [0.2, 0.0],
    "mass": 1.0,
    "name": "Extra"
  }
}
```

Notes:

- Adding a body changes the system from 3‑body to \(N\)-body, still using the same small‑N engine.
- Accelerations are updated for the new body, but existing bodies’ cached accelerations only become consistent after the next `step()` (which is fine for real‑time viewing).

### Compute velocity (handled by controller)

The viewer may send a request like:

```json
{"type": "compute_velocity", "data": {"position": [x, y, z], "mass": 1.0, "requestId": "ui"}}
```

In the unified backend, this is handled specially: the controller calls `calculate_velocity(position, mass)` and returns a `velocity_computed` message.

In Ethan’s simulator, `calculate_velocity(...)` currently returns a placeholder `[0.0, 0.0]`.

### Reset (optionally switch scenario)

```json
{"type": "reset", "data": {"scenario": "three_body"}}
```

Reset:

- Re-initializes the simulator from scratch using the scenario factory
- Clears queued states
- Marks the next streamed state with `reset_occurred = true` internally (controller may tag init messages)

---

## How to run it locally

From the repo root:

```bash
cd unified
pip install -r backend/requirements.txt
uvicorn backend.controller:app --host 0.0.0.0 --port 8000
```

Then open the dashboard and launch Ethan’s sim:

- Dashboard: `http://localhost:8000`
- Real‑time viewer: `http://localhost:8000/viewer.html#realtime&sim=ethansim`

To force the scenario via URL (if the viewer supports it in your version), you want:

- sim id: `ethansim`
- scenario: `three_body`

The backend websocket is:

- `/ws?sim_id=ethansim&scenario=three_body`

---

## Debugging tips (quick)

- **If motion “explodes” immediately**: lower `dt`, increase `softening` slightly.
- **If it looks too “mushy” / bodies pass through unrealistically**: decrease `softening`.
- **If the viewer stutters**: the sim loop runs at a fixed physics FPS; state queue drops frames if the client can’t keep up (expected).
- **If you changed the scenario list and it doesn’t show up**: confirm `SIMULATION_REGISTRY["ethansim"]["scenarios"]` includes it.

---

## Extending the 3‑body simulation

Common additions:

- **Energy / angular momentum diagnostics**:
  - Add functions to compute kinetic + potential energy and track drift over time.
- **More scenario presets**:
  - Add a new factory like `create_figure_eight_three_body()` and a new entry in `SCENARIOS`.
- **Better `calculate_velocity`**:
  - Implement a helper that returns a circular‑orbit velocity around the barycenter for interactive body placement.

Where to implement:

- Add scenario factories + config in `unified/simulations/ethan/app.py`
- Add UI/controls in `unified/frontend/viewer.html` (viewer side) if you want new commands/knobs

