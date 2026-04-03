# Gravity Simulation — Implementation Overview

This document describes the current simulation architecture for the **Gravitational Simulation of Solar System Formation** project. It references actual modules under `src/gravity/`. For a short explanation of every UI/CLI input, see [SIMULATION_TUNING.md](SIMULATION_TUNING.md#input-reference-what-each-control-does).

## Physical model

- **Newtonian gravity** between all pairs of point masses.
- **Central star:** implemented as particle index 0 with large mass `M_star`.
- **Softening:** distance in force law is `sqrt(r² + ε²)` so forces stay finite at small separations.
- **Optional dark matter halo:** static Hernquist profile centred at the origin adds acceleration **a = −G M_halo / (r + a_halo)²** (radially inward). This gives a flatter rotation curve and helps maintain disk orbits; initial circular velocities are set using enclosed mass (star + halo).
- **Units:** mass, distance, and time are in code units (e.g. G = 1); we do not convert to physical units.
- **2D / 3D:** positions and velocities are (x, y) or (x, y, z). No periodic boundaries; particles move in open space.

## Key equations

- **Particle–particle acceleration** on particle i:  
  **a_i = G Σ_{j≠i} m_j (r_j − r_i) / (|r_j − r_i|² + ε²)^(3/2)**
- **Halo acceleration** (if M_halo > 0): **a_halo(r) = −G M_halo / (r + a_halo)²** (unit vector toward origin). Total acceleration is particle term + halo term.
- **Circular orbit speed** at radius r: **v = sqrt(G M_enc(r) / r)** where M_enc is enclosed mass (star + Hernquist enclosed halo mass at r). Used to set initial disk/cloud velocities.
- **Integration:** Leapfrog (kick-drift-kick) for stability and approximate energy conservation.

## Module roles

| Module | Role |
|--------|------|
| `state.py` | `ParticleState`: positions (N,2) or (N,3), velocities, masses. Central star is index 0. |
| `forces_cpu.py` | `compute_accelerations()` (loop), `compute_accelerations_vectorized()` (NumPy), and `compute_halo_acceleration(positions, M_halo, a_halo)` for the Hernquist halo. |
| `forces_gpu.py` | GPU (CuPy) version of vectorized forces and `compute_halo_acceleration` (CPU). |
| `init_conditions.py` | `make_disk_2d`, `make_cloud_2d`, `make_disk_3d`, `make_cloud_3d` — all accept `M_halo`, `a_halo` and set v_circ from enclosed mass (star + halo). |
| `integrators.py` | `leapfrog_step(state, dt, accel_fn)` and `euler_step()` for testing. |
| `collisions.py` | Optional: `resolve_collisions(state, r_collide, star_index=0)` — inelastic mergers; returns state with smaller N. |
| `diagnostics.py` | Kinetic/potential energy, angular momentum, `SimulationLog`. |
| `demo_2d.py` / `demo_3d.py` | Main loop: build ICs, run Leapfrog (accel = particle forces + halo if M_halo > 0), optional collisions, replay export. CLI: n, steps, dt, softening, M_star, m_particle, r_min, r_max, `--M-halo`, `--a-halo`, `--collisions`, `--r-collide`, `--gpu`. |

## Initial conditions

- **Disk:** Particles in an annulus [r_min, r_max]. Tangential velocity set to **v_circ = sqrt(G M_enc(r) / r)** where M_enc includes the star and the Hernquist enclosed halo mass at r; small random perturbations are added.
- **Cloud:** Particles randomly inside r_max; velocities use a fraction of v_circ to control angular momentum.

Particle 0 is the central star (mass M_star, at origin, zero velocity). Disk/cloud particles have mass `m_particle` if set, otherwise `1/(N+1)`. The **mass ratio** (star mass / particle mass) is the main knob for stability: high ratio (e.g. 1000:1 or 10000:1) keeps the potential dominated by the star and helps orbits persist.

When **collisions** are enabled (`--collisions`), after each integration step `resolve_collisions` merges particle–star and particle–particle pairs within `r_collide`. Replay files can then have variable particle count per snapshot; see `docs/REPLAY_FORMAT.md`.

## Diagnostics

- **Total kinetic energy** K = (1/2) Σ m_i v_i²  
- **Total potential energy** U = −G Σ_{i<j} m_i m_j / r_ij (softened)  
- **Total angular momentum** L = Σ m_i (r_i × v_i) (z-component in 2D)

These are logged over time via `SimulationLog` and can be plotted to check conservation.

## Running

- Tests: `cd src && python -m gravity.tests_2d`
- 2D demo: `cd src && python -m gravity.demo_2d`  
  Optional: `--n`, `--steps`, `--dt`, `--ic disk|cloud`, etc., as implemented in `demo_2d.py`.
