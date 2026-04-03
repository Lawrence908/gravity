# Simulation Tuning — Clumps and Long-Term Orbits

Notes on getting desirable behavior: initial clumping, orbits that persist, and avoiding everything spiraling into a tight cluster around the star.

---

## Input reference (what each control does)

| Input | What it does |
|-------|----------------|
| **Spatial dimension** | 2D = disk in a plane; 3D = thickened disk in space. |
| **Initial conditions** | **Disk:** annulus between r_min and r_max with near-circular orbits. **Cloud:** random positions inside r_max with partial angular momentum. |
| **Number of particles** | Total disk/cloud particles (plus one central star). 1000–5000 is typical for spiral-galaxy-style runs. |
| **Time steps** | How many integration steps to run. More steps = longer simulation time. |
| **Time step size (Δt)** | Integration step in code units. Smaller = more accurate but slower; use ~0.01–0.02 for stability. |
| **Snapshot every N steps** | How often to save a frame for the replay. Larger N = smaller replay file. |
| **Mass ratio (star : particle)** | Single slider: star mass is fixed at 1; the slider sets how much heavier the star is than each particle (e.g. 1000:1 → m_particle = 0.001). Higher ratio = lighter particles = more stable orbits. Sphere sizes in the viewer scale with mass. |
| **Min initial radius (r_min)** | Inner edge of the disk annulus. Keep > 0.5 so particles don’t start on plunging orbits. |
| **Max initial radius (r_max)** | Outer edge; particles start within this radius. Larger = start further out, longer orbital times. |
| **Gravitational softening (ε)** | Softens gravity at small separations so force doesn’t blow up. Larger ε = weaker pull at close range; try 0.05–0.1. |
| **Dark matter halo mass** | Optional Hernquist halo centred at origin. 0 = off. Non-zero adds extra inward pull so the disk can maintain a flatter rotation curve (spiral-galaxy-like). |
| **Halo scale radius** | Scale length of the dark matter halo; larger = more spread-out halo. |
| **Random seed** | Fixes the random layout of particles for reproducible runs. |
| **Run name** | Name for the saved replay; leave empty for an auto-generated name. |
| **Use GPU** | Use CuPy/CUDA for force computation (faster for large N). |
| **Enable collisions** | Inelastic mergers: particles within collision radius merge. Off = particles pass through each other. |
| **Collision radius** | Distance below which two bodies merge. Default (auto) ≈ 1× softening; keep small to avoid over-merging. |

In the **viewer**, sphere sizes are drawn proportional to mass (cube-root of mass ratio) so you can see the star vs particle mass by eye.

---

## Softening: more or less attraction?

**Increasing gravitational softening (ε) makes particles *less* attracted to each other at very close distances.**

- The force uses a softened distance: \( r_{\mathrm{eff}}^2 = r^2 + \varepsilon^2 \), so the acceleration is bounded and never infinite.
- **Larger ε** → force is weaker when \( r \lesssim \varepsilon \). Close encounters are “smoothed out”; particles are less violently deflected or pulled into mergers.
- **Smaller ε** → force approaches pure Newtonian at close range; stronger attraction when particles get very close, which can cause numerical issues and rapid inspirals.

So: **more softening = less attraction at close range**. That can help clumps survive close passes and reduce the tendency for everything to merge into one central blob. Try ε in the 0.05–0.1 range; going too large will make the potential too soft and orbits less Keplerian.

## Starting further out: r_max and r_min

- **Max initial radius (r_max):** Larger values place the disk (or cloud) further from the star. With particles starting at larger r, they have more room to form clumps before strong central pull dominates, and orbital times are longer. **Try r_max = 4–6** (or more) instead of 2 for “start further out” runs.
- **Min initial radius (r_min), disk only:** Raising r_min keeps the *inner* edge of the annulus away from the star. That avoids placing many particles on initially tight orbits that quickly “crash past” the star and spiral in. Example: **r_min = 1.0, r_max = 5.0** for a wider, more distant annulus.

Together, larger r_max and (for disk) a higher r_min give particles more distance and time to clump and to maintain orbits before being dragged inward.

## Other knobs

- **Mass ratio:** Use the mass-ratio slider so the star dominates (e.g. 1000:1 or 10000:1). If the total disk mass is comparable to the star, the disk is violently unstable and particles spiral in quickly.
- **Dark matter halo:** Adding M_halo > 0 gives extra centripetal support and a flatter rotation curve; helps keep outer particles in orbit and can produce more spiral-galaxy-like structure.
- **Time step (Δt):** Smaller dt improves energy conservation; use ~0.01–0.02. Reduce if you see wild energy drift or ejections.
- **Steps:** For long-term orbits, run 5000–50000+ steps so you can see whether clumps persist or merge into the core.
- **Initial conditions:** “Disk” with circular-ish velocities gives ordered motion; “cloud” is more chaotic. Disk is the usual choice for spiral-like runs.
- **Collisions:** On = explicit mergers within r_collide (keep r_collide small, ~1× softening). Off = particles pass through; gravity alone still causes clumping.

## Suggested starting point for “clumps + long-term orbits”

- **n:** 2000–5000  
- **r_max:** 4–6 (particles start further out)  
- **r_min:** 1.0–2.0 (inner edge away from star)  
- **ε (softening):** 0.05–0.08  
- **Mass ratio:** 1000:1 or higher (star dominates)  
- **M_halo:** 0 to start; try 5–20 for spiral-galaxy-style runs  
- **a_halo:** ~0.5× r_max (e.g. 5 if r_max ≈ 10)  
- **steps:** 5000+ (e.g. 20000–50000 for long runs)  
- **dt:** 0.01 (or 0.02 if orbits look stable)

Then adjust r_max, r_min, mass ratio, and halo based on whether clumps stay coherent and orbit longer or still spiral in too quickly.
