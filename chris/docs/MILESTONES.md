# Milestones

High-level milestones aligned with `docs/project-outline.md` and the four-phase plan.

## Phase 1 — 2D prototype (CPU)

- **Bootstrap**
  - Repo and environment; `cd src && python -m main` runs.
- **Simulation core**
  - Gravitational force model with softening (loop + vectorized).
  - Leapfrog (or Velocity Verlet) integrator; adjustable timestep.
  - Central mass (star) and particle disk / cloud initial conditions.
- **Diagnostics**
  - Track total kinetic and potential energy and angular momentum.
  - Log summaries; optional plots of conservation trends.
- **Visualization**
  - Live 2D scatter with central star and particles; optional frame export to `outputs/frames/`.
- **Validation**
  - Stability over extended runs; approximate energy and angular momentum conservation; orbital behavior looks plausible.

## Phase 2 — 3D extension

- Extend particle state and force calculation to 3D.
- Improve visualization and camera control.
- Increase particle count where feasible.

## Phase 3 — Scaling and optimization

- Vectorize and optimize CPU force calculation; profile and benchmark.
- Target 10k–50k then 50k–100k+ particles.
- Optional: GPU acceleration (e.g. Numba, CuPy) after correctness is solid.

## Phase 4 — Web-based visualization

- Define replay file format; export positions (and metadata) at fixed intervals.
- Build WebGL viewer: load replay, 3D render, camera and timeline controls.
- Heavy computation remains offline; web app for replay and interaction.

## Scientific analysis and presentation

- Compare behavior to theory (e.g. circular orbits, conservation).
- Document assumptions and limitations relative to real solar system formation.
- Prepare presentation and demo for the class.
