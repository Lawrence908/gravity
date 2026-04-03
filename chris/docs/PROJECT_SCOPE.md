# Project Scope

## Core goal

Create an **educational simulation** that demonstrates how gravity can organize matter into stable orbital structures. We model a simplified system: a central mass (star) and many small particles (disk or cloud initial conditions), using **Newtonian gravity only**. The result is a visual, interactive tool to observe emergent structure and to discuss limitations relative to real solar system formation.

## In scope

- Newtonian gravitational force between all particles (point masses).
- Gravitational softening to avoid numerical singularities.
- Central star (one dominant mass) and configurable particle count and distribution.
- **Initial conditions:** rotating particle disk and optional random particle cloud; configurable angular momentum.
- Stable time integration (Leapfrog or Velocity Verlet); adjustable timestep.
- **Diagnostics:** total kinetic and potential energy, angular momentum; stability monitoring and logging.
- 2D prototype (Phase 1); later 3D extension, scaling (10k–100k+ particles), and WebGL replay viewer.
- Clear documentation of assumptions and limitations.

## Out of scope

- Gas dynamics, magnetic fields, radiation.
- Star formation, hydrodynamics, feedback.
- Relativistic effects.
- Realistic units or calibration to physical solar system parameters (we use code units for clarity).

We state these exclusions explicitly and discuss how they affect interpretation.

## Constraints

- Educational accuracy and conceptual clarity over experimental precision.
- Stable, demonstrable runs over maximum realism.
- CPU-first implementation; GPU and advanced optimizations only after correctness is established.

## Success criteria (from project-outline)

We consider the project successful if:

- The simulation runs stably over long timescales.
- Emergent structure is clearly visible (e.g. disk evolution, bounded orbits).
- The 3D visualization (when implemented) is intuitive and interactive.
- We can clearly explain both the physics and the limitations.
- The class gains insight from the visual demonstration.
