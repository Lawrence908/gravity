# Development Strategy: Gravitational Simulation of Solar System Formation

## Project focus

Single project: a **gravitational simulation of solar system formation**. We model a central star and many particles (disk or cloud initial conditions) under Newtonian gravity, with the goal of observing how orbital structure emerges and explaining both the physics and the limitations.

## Four-phase plan

Development follows the computational plan in `docs/project-outline.md`:

### Phase 1 — 2D prototype (CPU)

- Implement gravity (Newtonian + softening) and numerical integration (Leapfrog or Velocity Verlet) in 2D.
- Implement central mass and particle disk / cloud initial conditions.
- Add diagnostics: total kinetic and potential energy, angular momentum.
- Validate: stable orbits, conservation behavior, bounded motion.
- Deliver: stable 2D CPU simulation with configurable parameters and basic 2D visualization.

### Phase 2 — 3D extension

- Extend particle state and force calculation to 3D.
- Improve visualization and camera control.
- Begin increasing particle count where feasible.

### Phase 3 — Scaling and optimization

- Increase particle count toward 10k–50k (early) and 50k–100k+ (mid/later).
- Vectorize force calculations (NumPy); profile and benchmark.
- Optionally explore GPU (e.g. Numba, CuPy) only after correctness is established.

### Phase 4 — Web-based visualization

- Export simulation data (positions over time) in a defined replay format.
- Build a client-side WebGL viewer to load and replay data with camera and timeline controls.
- Computation stays offline; the web app is for interaction and presentation.

## Team roles (4 members)

See `docs/work-breakdown.md` for full detail. Summary:

- **Physics implementation lead:** Force model, integrator, initial conditions, stability tuning.
- **Diagnostics and validation:** Energy and angular momentum tracking, stability analysis, parameter experiments, documentation.
- **Performance and scaling:** CPU optimization, profiling, GPU exploration if pursued, replay export.
- **Visualization and presentation:** 3D extension, replay format support, WebGL viewer, presentation and demo.

All members contribute to scientific interpretation, model limitations, and final presentation.

## Principles

- Correctness and clarity before optimization.
- Educational accuracy and honest discussion of simplifications.
- Stable, demonstrable runs over maximum realism.
