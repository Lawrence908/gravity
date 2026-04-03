# Gravitational Simulation of Solar System Formation

An educational ASTR 311 project: a visual simulation showing how gravity organizes matter into stable orbital structures. We model a simplified system in which many small particles interact with a central mass (a star), so we can observe how structure emerges through gravitational attraction alone.

## What this project is

We build a Newtonian N-body gravity simulation with:
- A central star and orbiting particles (disk or cloud initial conditions)
- Leapfrog integration and gravitational softening
- Energy and angular momentum diagnostics
- A 2D prototype (Phase 1), with plans for 3D, scaling, and a WebGL replay viewer

The goal is not full astrophysical realism but a clear, educational model that illustrates how gravity shapes motion and structure in space.

## Course context

Upper-level astronomy group project. The simulation connects lecture concepts (Newtonian gravity, orbital motion, multi-body dynamics, emergent structure) to concrete, inspectable computations.

- **Team:** 4 members (see `docs/work-breakdown.md` for roles)
- **Course / term:** _TBD_
- **Instructor:** _TBD_

## Development phases

See `docs/project-outline.md` and `docs/DEVELOPMENT_STRATEGY.md` for the full plan.

1. **Phase 1 — 2D prototype (CPU):** Gravity, integration, disk/cloud ICs, diagnostics, validation.
2. **Phase 2 — 3D extension:** 3D physics and visualization.
3. **Phase 3 — Scaling:** Increase particle count (10k–100k+), CPU optimization, optional GPU.
4. **Phase 4 — WebGL viewer:** Export replay data; browser-based 3D viewer for replay and interaction.

## Getting started

- **Setup:** `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`).
- **Run tests:** From project root, `cd src && python -m gravity.tests_2d`
- **Run 2D demo:** `cd src && python -m gravity.demo_2d` (optional args: `--n`, `--steps`, `--ic`, `--dt`)

See `docs/PROJECT_SCOPE.md` and `docs/ARCHITECTURE.md` for scope and code layout.
