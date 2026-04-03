# Project Context

This document summarizes the agreed project plan and how the repo is organized.

## Project title

**Gravitational Simulation of Solar System Formation** (ASTR 311 group project).

## Goal

Build a visual simulation that demonstrates how gravity can organize matter into stable orbital structures over time. We model a simplified system: a central mass (star) and many small particles (disk or cloud initial conditions), using Newtonian gravity only. The result is educational—we observe how structure emerges from gravity alone and discuss limitations relative to real planet formation.

## Core assumptions

- Newtonian mechanics; point masses; no gas, magnetic fields, or radiation.
- Gravitational softening to avoid numerical singularities.
- Emphasis on visual clarity and conceptual understanding.

## Team and roles (4 members)

See `docs/work-breakdown.md` for the full breakdown. In brief:

- **Member 1 – Physics implementation:** Force model, integrator, initial conditions, stability.
- **Member 2 – Diagnostics and validation:** Energy and angular momentum tracking, stability analysis, documentation.
- **Member 3 – Performance and scaling:** CPU optimization, profiling, GPU exploration, replay export.
- **Member 4 – Visualization and presentation:** 3D extension, replay format support, WebGL viewer, presentation.

All contribute to scientific interpretation, model limitations, and final presentation.

## Development phases

1. **Phase 1 – 2D prototype (CPU):** Gravity, integration, disk/cloud ICs, diagnostics, 2D visualization. Target: stable 2D run with 10k–20k particles (or smaller for quick demos).
2. **Phase 2 – 3D extension:** 3D state and forces, improved visualization and camera.
3. **Phase 3 – Scaling and optimization:** Higher particle counts (50k–100k+), vectorization, optional GPU (e.g. Numba/CuPy).
4. **Phase 4 – Web-based visualization:** Replay file format and WebGL viewer for interactive replay.

Heavy computation is done offline; the web component is for replay and exploration.

## Hardware (VM)

- **GPU:** NVIDIA RTX A2000, 12 GB VRAM, CUDA 13.1
- **CPU:** Xeon E5-2643 v4 @ 3.40 GHz, 24 logical cores
- **RAM:** 31 GB
- **OS:** AlmaLinux 10.1

## Key docs

- **docs/project-outline.md** – Overview, concepts, phases, success criteria, VM specs.
- **docs/work-breakdown.md** – Detailed work breakdown and member roles.
- **docs/DEVELOPMENT_STRATEGY.md** – Phase-by-phase development approach.
- **docs/PROJECT_SCOPE.md** – In-scope / out-of-scope, success criteria.
- **docs/ARCHITECTURE.md** – Source layout and modules.
- **docs/PHYSICS_NOTES.md** – Equations, softening, disk ICs, conservation.
- **docs/MILESTONES.md** – Phase milestones.
- **docs/TASKS.md** – Phase 1 task list.
