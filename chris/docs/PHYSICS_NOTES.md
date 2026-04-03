# Physics Notes — Gravitational Simulation of Solar System Formation

Brief physics reference for the N-body gravity simulation. For **what each simulation input does**, see [SIMULATION_TUNING.md](SIMULATION_TUNING.md#input-reference-what-each-control-does).

## Model

- **Newtonian gravity** between point masses.
- **Central star:** one particle (index 0) with mass M_star at the origin; all other particles are disk/cloud particles with mass m_particle.
- **Gravitational softening:** we use a softened distance so forces do not diverge at small separations.
- **Optional dark matter halo:** a static Hernquist halo can be added (M_halo, a_halo); it contributes extra inward acceleration and is used when setting initial circular velocities.

## Force law

Acceleration on particle i due to all others:

**a_i = G Σ_{j≠i} m_j (r_j − r_i) / (|r_j − r_i|² + ε²)^(3/2)**

- G: gravitational constant (e.g. 1 in code units).
- ε: softening length (e.g. 0.05); prevents singularities when particles get very close.
- In 2D, r and v are (x, y); the formula is the same with 2-vectors.

## Circular orbits and halo

For a particle at distance r, the circular orbital speed is **v_circ = sqrt(G M_enc(r) / r)** where M_enc is the enclosed mass. With no halo, M_enc = M_star. With a Hernquist halo, M_enc = M_star + M_halo × r²/(r + a_halo)². Initial disk/cloud velocities use this v_circ (with small random perturbations). In 2D, angular momentum per unit mass is L_z = x v_y − y v_x (out of the plane).

## Integration

- **Leapfrog (kick-drift-kick)** is used for stability and approximate energy conservation:
  - Half-kick: v → v + (dt/2) a
  - Drift: r → r + dt v
  - Half-kick: v → v + (dt/2) a_new
- **Euler** is available for quick tests but is not suitable for long runs.
- Timestep dt must be small enough that orbits remain stable (e.g. fraction of the shortest dynamical time).

## Conservation (ideal case)

- **Total energy** E = K + U should be approximately constant (Leapfrog preserves it well in practice).
  - Kinetic: K = (1/2) Σ m_i v_i²  
  - Potential: U = −G Σ_{i<j} m_i m_j / r_ij (with softened r_ij).
- **Total angular momentum** L = Σ m_i (r_i × v_i) is exactly conserved by Newtonian gravity (in 2D, only the z-component is non-zero). We track it to validate the integrator and initial conditions.

## Initial conditions

- **Disk:** Particles in an annulus [r_min, r_max] with tangential velocity ≈ v_circ(r) (v_circ includes star + halo enclosed mass); small random perturbations.
- **Cloud:** Particles inside r_max; velocities use a fraction of v_circ for partial angular momentum.

Both include the central star as particle 0 (mass M_star, at origin, zero velocity). The **mass ratio** (M_star / m_particle) should be large (e.g. 1000:1 or more) so the star dominates and orbits are stable.

## Collisions (optional)

When collisions are enabled (e.g. `--collisions`), the simulation resolves **inelastic mergers** each step:

- **Collision radius** r_collide: any pair with separation &lt; r_collide is merged. Default is **1× softening** so only genuinely close pairs merge. A larger r_collide (e.g. 2× softening) causes rapid over-merging in dense disks: most particles merge away in few steps and the file becomes very small with few bodies; keep it small for gradual “chunking” and many particles.
- **Particle–star:** a particle within r_collide of the star is absorbed (star mass increases, particle removed). Star position/velocity unchanged (toy: star dominates).
- **Particle–particle:** the pair is merged into one body at the centre of mass, with momentum-conserving velocity; one particle is removed.
- **Mass and momentum** are conserved at each merge; **total energy** is not (inelastic). Angular momentum is conserved. For a stable central attractor and “hugging” clumps, use **star mass much larger than particle mass** (e.g. M_star=1, m_particle=0.001).

## What we do not include

- Gas dynamics, magnetic fields, radiation.
- Relativistic effects.
- Fragmentation (splitting of bodies).

Collisions/mergers are optional; when disabled, we do not include collisions or fragmentation. We explicitly discuss these simplifications and how they differ from real planetary formation.
