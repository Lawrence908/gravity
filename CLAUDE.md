# CLAUDE.md — Gravity Simulation Repository Guide

This file provides guidance for AI assistants working in this repository.

## Project Overview

**Cosmic Origins** is an educational ASTR 311 group project implementing Newtonian N-body gravitational simulations in Python. It demonstrates how gravity organizes matter into orbital structures (protostellar disk → planetary system).

The repository has two sub-projects:

| Directory | Purpose |
|-----------|---------|
| `chris/` | Original 2D/3D CPU/GPU gravity simulation (standalone) |
| `unified/` | Multi-user collaborative web platform (4 team members' simulators) |

## Repository Structure

```
gravity/
├── chris/                          # Chris's standalone simulation
│   ├── src/gravity/               # Core physics engine (~2,400 LOC)
│   │   ├── state.py               # ParticleState dataclass
│   │   ├── forces_cpu.py          # CPU O(N²) gravity + Hernquist halo
│   │   ├── forces_gpu.py          # GPU acceleration (CuPy/Numba stub)
│   │   ├── integrators.py         # Euler and Leapfrog integrators
│   │   ├── init_conditions.py     # Disk/cloud initial condition generators
│   │   ├── diagnostics.py         # Energy, angular momentum, SimulationLog
│   │   ├── collisions.py          # Inelastic particle mergers
│   │   ├── demo_2d.py             # 2D runner: CLI, live viz, replay save
│   │   ├── demo_3d.py             # 3D extension demo
│   │   ├── tests_2d.py            # Unit tests: symmetry, CoM, energy drift
│   │   ├── tests_3d.py            # Unit tests: 3D orbits, AM conservation
│   │   ├── tests_collisions.py    # Unit tests: particle mergers
│   │   ├── viz_live.py            # Matplotlib live 2D visualization
│   │   ├── viz_export.py          # PNG frame export for animations
│   │   ├── replay.py              # Save/load snapshots (.npz, JSON)
│   │   ├── progress.py            # CLI progress reporting
│   │   └── benchmark.py           # Performance profiling
│   ├── web-viewer/                # Static HTML replay browser
│   ├── tools/                     # sim_server.py, export, thin_replay utilities
│   ├── docs/                      # 11 markdown documentation files
│   ├── pyproject.toml             # Build config, Python >=3.10
│   ├── requirements.txt           # numpy, matplotlib, cupy-cuda12x
│   ├── Dockerfile                 # nvidia/cuda:12.2.2-devel base, port 8117
│   ├── compose.yaml               # GPU passthrough, health checks
│   └── INSTRUCTIONS.md            # AI assistant guidelines for this sub-project
│
└── unified/                       # Collaborative web platform
    ├── backend/
    │   ├── controller.py          # FastAPI app + WebSocket (~444 LOC)
    │   └── requirements.txt       # fastapi, uvicorn, numpy, matplotlib, msgpack
    ├── frontend/
    │   ├── index.html             # Dashboard with simulation picker
    │   ├── viewer.html            # Replay + real-time viewer
    │   └── replays/               # Pre-computed .json replay files
    ├── core/
    │   ├── gravity/               # Shared physics engine (mirrors chris/src/gravity)
    │   ├── common/                # Shared utilities (logging)
    │   └── config.py              # GravityConfig + AppConfig dataclasses
    ├── simulations/
    │   ├── chris/app.py           # CPU leapfrog N-body (AsyncSimulator)
    │   ├── ethan/app.py           # 3-body problem + Pluto system
    │   ├── jasper/app.py          # GPU Yoshida 4th-order + GR corrections
    │   └── brad/app.py            # Binary star template
    ├── Dockerfile                 # python:3.12-slim, port 8124
    └── compose.yaml               # Health checks, frontend/replays bind mount
```

## Development Commands

### Chris's Standalone Project

```bash
cd chris

# Install dependencies
pip install -r requirements.txt
# or with uv:
uv pip install -r requirements.txt

# Run tests
cd src
python -m gravity.tests_2d
python -m gravity.tests_3d
python -m gravity.tests_collisions

# Run 2D demo
python -m gravity.demo_2d                          # defaults
python -m gravity.demo_2d --n 1000 --steps 2000   # custom
python -m gravity.demo_2d --ic disk                # disk IC
python -m gravity.demo_2d --ic cloud               # cloud IC
python -m gravity.demo_2d --dt 0.01                # timestep

# Run 3D demo
python -m gravity.demo_3d

# Benchmark
python -m gravity.benchmark
```

### Unified Platform

```bash
cd unified

# Install backend dependencies
pip install -r backend/requirements.txt

# Run development server (from unified/)
uvicorn backend.controller:app --host 0.0.0.0 --port 8000 --reload

# Or with Docker
docker compose up --build
# Frontend available at http://localhost:8124
# Health check: http://localhost:8124/health
```

### Docker (Chris's project)

```bash
cd chris
docker compose up --build
# Server available at http://localhost:8117
```

## Code Conventions

### Naming

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase` (e.g., `ParticleState`, `SimulationLog`, `GravityConfig`)
- **Private/internal**: underscore prefix (e.g., `_make_accel_fn`, `_hernquist_enclosed`)
- **Constants**: `UPPER_CASE` (e.g., `G_DEFAULT`, `SCENARIOS`)

### Type Hints

The core gravity packages (`chris/src/gravity` and `unified/core/gravity`) use `from __future__ import annotations` and full type annotations throughout. Elsewhere in the repo, prefer that style when modifying or adding code. Numpy array shapes are documented in docstrings where relevant (e.g., `"shape (N, 2) or (N, 3)"`).

### Docstrings

NumPy-style docstrings with section headers such as `Parameters`, `Returns`, and `Examples` (using underlined section separators where applicable) are the preferred convention going forward, matching the core gravity modules. Inline physics comments explain equations (softening, leapfrog, Hernquist potential).

### Key Abstractions

- **`ParticleState`** (`state.py`): Central dataclass holding `pos`, `vel`, `mass` arrays
- **`AccelerationFn`** protocol: Callable `(state: ParticleState) -> np.ndarray` — used for swappable force implementations
- **`AsyncSimulator`** (unified): Base interface all team simulators must implement

### Adding a New Simulator (Unified Platform)

1. Create `unified/simulations/<name>/app.py`
2. Implement the `AsyncSimulator` interface (see `backend/controller.py` for the registration pattern)
3. Register the simulator in `unified/backend/controller.py`
4. Add replay JSON files to `unified/frontend/replays/` if applicable

## Physics Model

- **Integrator**: Leapfrog (symplectic, energy-conserving) preferred; Euler available for comparison
- **Force law**: Newtonian gravity with softening: `F = G*m1*m2 / (r² + ε²)^(3/2)`
- **Softening parameter ε**: Prevents singularities at close range; tune via `GravityConfig.softening`
- **Hernquist halo**: Optional static background potential for galaxy disk stability
- **Diagnostics**: Total energy (KE + PE) and angular momentum are tracked and should conserve within acceptable drift
- **Collision merging**: Inelastic mergers conserving momentum and volume (`collisions.py`)

Key conservation law tolerances from tests:
- Energy drift: < 1% over simulation duration
- Angular momentum: conserved to machine precision (leapfrog)

## Configuration

`GravityConfig` fields in `unified/core/config.py`:

| Field | Default | Description |
|-------|---------|-------------|
| `dim` | varies | Spatial dimensionality of the simulation |
| `n_particles` | varies | Number of simulation bodies |
| `time_step` | varies | Integration timestep |
| `softening_length` | varies | Gravitational softening length ε |
| `M_star` | varies | Central star mass |
| `r_min` | varies | Minimum radius used for initialization |
| `r_max` | varies | Maximum radius used for initialization |
| `ic_type` | varies | Initial-condition profile/type selector |
| `M_halo` | varies | Halo mass parameter |
| `a_halo` | varies | Halo scale-length parameter |

## Replay Format

Replays are saved as JSON with the following structure:

```json
{
  "metadata": { "n_particles": 100, "steps": 500, "dt": 0.01, ... },
  "frames": [
    { "positions": [[x,y], ...], "velocities": [[vx,vy], ...], "masses": [...] },
    ...
  ]
}
```

Binary snapshots use `.npz` format (numpy compressed). See `chris/docs/REPLAY_FORMAT.md` for full spec.

## API Endpoints (Unified Platform)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve frontend dashboard |
| GET | `/health` | Health check |
| GET | `/api/simulations` | List available simulators |
| GET | `/api/replays` | List pre-computed replays |
| GET | `/api/replays/{name}` | Download a replay JSON |
| WebSocket | `/ws?sim_id={sim_id}&scenario={scenario}` | Real-time simulation streaming |

## Docker Configuration

| Project | Base Image | Host Port | GPU |
|---------|-----------|-----------|-----|
| chris | `nvidia/cuda:12.2.2-devel-ubuntu22.04` | 8117 | Optional (compose override) |
| unified | `python:3.12-slim` | 8124 | No |

Environment variables used in containers:
- `PYTHONPATH`: `/app/src` (chris), `/app` (unified)
- `PORT`: `8000`
- `PYTHONUNBUFFERED`: `1`

## Educational Context

This is an ASTR 311 course project. When working on the physics:

- **Prioritize correctness and clarity** over computational realism
- **Use dimensionless units** — `G=1`, masses/distances normalized for numerical stability
- **Prefer leapfrog** over Euler for any simulation run > a few hundred steps
- **Test conservation laws** after any changes to forces or integrators
- **Document physics choices** with inline comments referencing the governing equations

Do NOT assume this needs to match experimental observational data. It is a pedagogical simulation.

## Gitignore

The following are excluded from version control:
- `.env` (secrets like `SIMLAB_PASSWORD`)
- `venv/`, `__pycache__/`
- `outputs/`, `replays/` (generated simulation files)
- `meetings/`

## Branch

Active development branch: `claude/add-claude-documentation-OioSx`
Main branch: `main`
