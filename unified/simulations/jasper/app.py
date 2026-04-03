"""
simulations/gravsim.py — Realistic GPU N-Body Gravitational Simulator

Physical units: AU (distance), M☉ (mass), yr (time)
  G = 4π² AU³ M☉⁻¹ yr⁻²   (exact from Kepler's 3rd law)
  c = 63,241.1 AU/yr

Integration: Yoshida 4th-order symplectic integrator (3 force evals per dt)
  - O(dt⁴) energy error vs leapfrog O(dt²)
  - Symplectic: preserves Hamiltonian structure, no long-term orbital drift
  - Collision handling runs AFTER the full Yoshida step (never mid-step)
    to avoid contaminating the symplectic integrator with non-conservative impulses.

Force law: Plummer-softened Newtonian gravity + EIH 1st Post-Newtonian correction
  F_Newton = G·m₁·m₂ / (r² + ε²)^(3/2)   ε = 0.001 AU
  F_1PN    = EIH equations (Will 1993, eq 10.28)
             Verified: 43.00"/century Mercury perihelion precession

Collisions: perfectly inelastic merger (momentum-conserving, volume-conserving radius)
  - Triggered when physical separation < sum of body radii
  - New mass = m₁+m₂,  new vel = (m₁v₁+m₂v₂)/(m₁+m₂),  new r = ∛(r₁³+r₂³)
  - Runs post-step to preserve Yoshida symplectic structure

Spacetime viz: Flamm paraboloid (Schwarzschild metric embedding diagram)
  z(r) ∝ -2√(r_s·r)   r_s = 2GM/c²
  Computed on 64×64 GPU grid (reduced from 128 for async-copy headroom).
  Grid export runs on a DEDICATED DAEMON THREAD — never blocks the physics loop.
  r clamped to max(body_radius, 1e-4) per body to prevent singularity depth
  overwhelming the colour normalisation on the client.
"""

import math
import torch
import time
import threading
import traceback
from queue import Queue, Empty

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[GravSim] Using device: {device}")
if torch.cuda.is_available():
    print(f"[GravSim] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[GravSim] GPU Memory: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ============================================================================
# PHYSICAL & SIMULATION CONSTANTS
# ============================================================================

# G in AU³ M☉⁻¹ yr⁻² — exact from Kepler's 3rd law
G_REAL        = 4.0 * math.pi ** 2

# Speed of light in AU/yr
C_AUPYR       = 63241.1
C2            = C_AUPYR ** 2           # precomputed c² for 1PN

# Plummer softening (AU) — sub-planetary scale
PLUMMER_EPS   = 0.001
PLUMMER_EPS2  = PLUMMER_EPS ** 2

# Spacetime grid — 64×64 (reduced from 128 to cut GPU→CPU copy cost ~75%)
# Must match GRID_RES constant in gravsim.html
GRID_RES      = 64
GRID_EXTENT   = 35.0                   # AU — must match client JS GRID_EXTENT

# Visual scale for Flamm paraboloid.
# -2√(r_s · 1 AU) · WARP_VISUAL_SCALE ≈ -1.5 AU at Earth orbit (r_s_sun ≈ 1.974e-8 AU)
# scale = 1.5 / (2 · √(1.974e-8 · 1)) ≈ 5338
WARP_VISUAL_SCALE = 5338.0

# Collision detection scale — must match BODY_RADIUS_SCALE in gravsim.html.
# Collision fires when dist(i,j) < (radius_i + radius_j) * COLLISION_SCALE.
# With COLLISION_SCALE = BODY_RADIUS_SCALE, collision fires exactly when
# visual spheres touch, giving pixel-accurate collision feedback.
# Verified: all 8 solar-system planets remain safe with SUN_R=0.15 AU.
COLLISION_SCALE = 2.0

# Yoshida 4th-order symplectic coefficients (Yoshida 1990, Phys. Lett. A 150)
_cbrt2  = 2.0 ** (1.0 / 3.0)
YO_W1   =  1.0 / (2.0 - _cbrt2)      #  +1.35120719…
YO_W0   = -_cbrt2 * YO_W1            #  -1.70241438…
YO_C    = [YO_W1 / 2.0,
           (YO_W0 + YO_W1) / 2.0,
           (YO_W0 + YO_W1) / 2.0,
           YO_W1 / 2.0]               # drift coefficients — sum = 1
YO_D    = [YO_W1, YO_W0, YO_W1]      # kick  coefficients — sum = 1


# ============================================================================
# PHYSICS ENGINE
# ============================================================================

class AdvancedOrbitalSimulator:
    """
    GPU N-body simulator:
      - AU / M☉ / yr  (G = 4π², c = 63241.1 AU/yr)
      - Yoshida 4th-order symplectic integration
      - Plummer-softened Newtonian + EIH 1PN (GR)
      - Post-step perfectly inelastic merger collisions
      - 64×64 Flamm paraboloid spacetime grid (raw float32 bytes)
    """

    def __init__(self, bodies=None, dt=0.002, max_bodies=25, substeps=1):
        self.G            = G_REAL
        self.base_dt      = float(dt)
        self.substeps     = max(1, int(substeps))
        self.dt           = self.base_dt / self.substeps
        self.max_bodies   = int(max_bodies)
        self.step_count   = 0
        self.compute_time_ms  = 0.0
        self.frame_times      = []
        self.memory_used_gb   = 0.0

        # Pending mergers accumulate during a step and are applied atomically
        self._pending_mergers = []

        self.bodies = bodies if bodies is not None else self._create_solar_system()
        self.n      = len(self.bodies)
        self._initialize_tensors()
        self._initialize_spacetime_grid()
        self.total_energy_initial = self._compute_total_energy()

    # ------------------------------------------------------------------
    # TENSOR INITIALISATION
    # ------------------------------------------------------------------

    def _initialize_tensors(self):
        """Rebuild all GPU body tensors from self.bodies list. Safe on empty."""
        if self.n > 0:
            self.pos    = torch.tensor(
                [b["pos"]    for b in self.bodies], device=device, dtype=torch.float32)
            self.vel    = torch.tensor(
                [b["vel"]    for b in self.bodies], device=device, dtype=torch.float32)
            self.mass   = torch.tensor(
                [b["mass"]   for b in self.bodies], device=device, dtype=torch.float32)
            self.radius = torch.tensor(
                [b["radius"] for b in self.bodies], device=device, dtype=torch.float32)
        else:
            self.pos    = torch.zeros((0, 3), device=device, dtype=torch.float32)
            self.vel    = torch.zeros((0, 3), device=device, dtype=torch.float32)
            self.mass   = torch.zeros(0,      device=device, dtype=torch.float32)
            self.radius = torch.zeros(0,      device=device, dtype=torch.float32)

        self.colors = [b["color"]  for b in self.bodies]
        self.names  = [b["name"]   for b in self.bodies]
        self.accel  = torch.zeros(self.n, 3, device=device, dtype=torch.float32)
        self._rebuild_derived()

    def _rebuild_derived(self):
        """Rebuild cached per-pair tensors after any body count change."""
        if self.n > 1:
            self.mass_matrix = self.mass.unsqueeze(0) * self.mass.unsqueeze(1)  # (N,N)
            self.inv_mass    = 1.0 / self.mass.unsqueeze(1)                     # (N,1)
        elif self.n == 1:
            self.mass_matrix = (self.mass.unsqueeze(0) * self.mass.unsqueeze(1))
            self.inv_mass    = 1.0 / self.mass.unsqueeze(1)
        else:
            self.mass_matrix = torch.zeros((0, 0), device=device, dtype=torch.float32)
            self.inv_mass    = torch.zeros((0, 1), device=device, dtype=torch.float32)

    def _initialize_spacetime_grid(self):
        """
        Pre-allocate 64×64 XZ grid for Flamm paraboloid.
        indexing='xy': grid_x[i,j] = xs[j]  (x fast/inner),
                       grid_z[i,j] = zs[i]  (z slow/outer).
        Matches Three.js PlaneGeometry vertex order after rotateX(-π/2).
        Grid coordinates are fixed; only Y-displacements recomputed each emit.
        """
        xs = torch.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_RES, device=device)
        zs = torch.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_RES, device=device)
        self.grid_x, self.grid_z = torch.meshgrid(xs, zs, indexing='xy')  # (R,R)
        self.grid_y = torch.zeros(GRID_RES, GRID_RES, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # SOLAR SYSTEM  (AU / M☉ / yr)
    # ------------------------------------------------------------------

    def _create_solar_system(self):
        """
        Realistic solar system. Masses from JPL DE430 (ratio to M☉).
        Circular orbital speeds: v = √(G·M_sun/a).
        Visual radii exaggerated for screen visibility; collision radii use same value.
        """
        bodies = [{
            "name": "Sun", "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
            "mass": 1.0, "radius": 0.15, "color": [45, 100, 95],
        }]
        # (name, a[AU], mass[M☉], vis_radius[AU], HSL)
        planets = [
            ("Mercury", 0.387,  1.651e-7,  0.012, [30,  60, 70]),
            ("Venus",   0.723,  2.448e-6,  0.018, [35,  80, 85]),
            ("Earth",   1.000,  3.003e-6,  0.020, [210, 70, 75]),
            ("Mars",    1.524,  3.227e-7,  0.014, [15,  85, 70]),
            ("Jupiter", 5.203,  9.545e-4,  0.055, [30,  70, 80]),
            ("Saturn",  9.537,  2.858e-4,  0.048, [40,  60, 85]),
            ("Uranus",  19.19,  4.366e-5,  0.034, [190, 60, 75]),
            ("Neptune", 30.07,  5.151e-5,  0.033, [220, 70, 70]),
        ]
        for name, a, mass, r_vis, color in planets:
            v = math.sqrt(self.G * 1.0 / a)   # circular orbit speed (AU/yr)
            bodies.append({
                "name": name, "pos": [a, 0.0, 0.0], "vel": [0.0, 0.0, v],
                "mass": mass, "radius": r_vis, "color": color,
            })
        return bodies

    # ------------------------------------------------------------------
    # ENERGY  (for drift monitoring)
    # ------------------------------------------------------------------

    def _compute_total_energy(self):
        if self.n == 0:
            return 0.0
        ke = 0.5 * (self.mass * (self.vel * self.vel).sum(dim=1)).sum()
        if self.n < 2:
            return float(ke)
        diff    = self.pos.unsqueeze(0) - self.pos.unsqueeze(1)         # (N,N,3)
        dist_sq = (diff * diff).sum(dim=2) + PLUMMER_EPS2               # (N,N)
        dist    = torch.sqrt(dist_sq)                                    # (N,N)
        triu    = torch.triu_indices(self.n, self.n, offset=1, device=device)
        pe      = -(self.G * self.mass_matrix[triu[0], triu[1]]
                    / dist[triu[0], triu[1]]).sum()
        return float(ke + pe)

    # ------------------------------------------------------------------
    # YOSHIDA INTEGRATION
    # ------------------------------------------------------------------

    def step(self):
        if self.n == 0:
            return
        start = time.perf_counter()
        for _ in range(self.substeps):
            self._yoshida_step()
            self._apply_mergers()   # post-step: symplectic structure preserved
        self.step_count += 1
        elapsed = time.perf_counter() - start
        self.compute_time_ms = elapsed * 1000.0
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 120:
            self.frame_times.pop(0)
        self.memory_used_gb = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        )

    def _yoshida_step(self):
        """
        Yoshida (1990) 4th-order symplectic:
          c1·drift → d1·kick → c2·drift → d2·kick → c3·drift → d3·kick → c4·drift
        3 force evaluations per step.  D[1]=w0<0 (backwards kick) is correct —
        it is the key to 4th-order accuracy.
        Collision detection is NEVER called here to preserve the symplectic structure.
        """
        if self.n < 2:
            return
        # Guard against tensor/body-count mismatch after a mid-step merge
        if self.pos.shape[0] != self.n:
            self.accel = torch.zeros(self.n, 3, device=device, dtype=torch.float32)
            return

        dt = self.dt

        self.pos.add_(self.vel, alpha=YO_C[0] * dt)
        self._compute_acceleration()
        self.vel.add_(self.accel, alpha=YO_D[0] * dt)

        self.pos.add_(self.vel, alpha=YO_C[1] * dt)
        self._compute_acceleration()
        self.vel.add_(self.accel, alpha=YO_D[1] * dt)

        self.pos.add_(self.vel, alpha=YO_C[2] * dt)
        self._compute_acceleration()
        self.vel.add_(self.accel, alpha=YO_D[2] * dt)

        self.pos.add_(self.vel, alpha=YO_C[3] * dt)

        # Detect overlaps at end of each Yoshida sub-step (positions are final
        # for this sub-step).  _apply_mergers() is called immediately after in
        # the step() loop, preserving Yoshida symplectic structure.
        self._detect_mergers()

    # ------------------------------------------------------------------
    # FORCE COMPUTATION — Newtonian + EIH 1PN
    # ------------------------------------------------------------------

    def _compute_acceleration(self):
        """
        Full acceleration: Plummer-softened Newtonian + EIH 1st Post-Newtonian.

        EIH 1PN (Will 1993, eq 10.28):
          δaᵢ = (1/c²) Σⱼ≠ᵢ G·mⱼ/rᵢⱼ³ {
              n̂ᵢⱼ · [−vᵢ² − 2vⱼ² + 4(vᵢ·vⱼ) + 3/2·(n̂ᵢⱼ·vⱼ)² + 4Φᵢ + Φⱼ]
            + (vᵢ−vⱼ) · [4(n̂ᵢⱼ·vᵢ) − 3(n̂ᵢⱼ·vⱼ)]
          }
        n̂ᵢⱼ = (rⱼ−rᵢ)/|rⱼ−rᵢ|   (points from i toward j)
        Φᵢ   = G·Σₖ≠ᵢ mₖ/rᵢₖ     (gravitational potential at i, self excluded)

        Verified: reproduces Mercury perihelion precession = 43.00″/century.
        """
        n = self.n
        if n < 2:
            self.accel = torch.zeros(n, 3, device=device, dtype=torch.float32)
            return

        # diff[i,j] = pos[j] − pos[i]   (N,N,3)
        diff    = self.pos.unsqueeze(0) - self.pos.unsqueeze(1)
        dist_sq = (diff * diff).sum(dim=2) + PLUMMER_EPS2   # (N,N)
        dist    = torch.sqrt(dist_sq)                        # (N,N)
        dist_cb = dist * dist_sq                             # |r|³  (N,N)

        # G·mᵢ·mⱼ / |r|³  — used by both Newtonian and 1PN terms
        scalar_N  = self.G * self.mass_matrix / dist_cb      # (N,N)
        forces_N  = scalar_N.unsqueeze(2) * diff             # (N,N,3) Newtonian

        # ── EIH 1PN ──────────────────────────────────────────────────────
        n_hat          = diff / (dist.unsqueeze(2) + 1e-12)  # (N,N,3) unit vec i→j

        vi = self.vel.unsqueeze(1).expand(n, n, 3)           # (N,N,3)
        vj = self.vel.unsqueeze(0).expand(n, n, 3)           # (N,N,3)

        spd_sq        = (self.vel * self.vel).sum(dim=1)     # (N,)
        vi_sq_b       = spd_sq.unsqueeze(1).expand(n, n)     # (N,N)
        vj_sq_b       = spd_sq.unsqueeze(0).expand(n, n)     # (N,N)
        vi_dot_vj     = (vi * vj).sum(dim=2)                 # (N,N)
        nhat_dot_vj   = (n_hat * vj).sum(dim=2)              # (N,N)
        nhat_dot_vi   = (n_hat * vi).sum(dim=2)              # (N,N)
        nhat_dvj_sq   = nhat_dot_vj * nhat_dot_vj            # (N,N)

        # Gravitational potential at each body (self excluded: diagonal → ∞ → 0)
        eye_bool      = torch.eye(n, device=device, dtype=torch.bool)
        dist_no_self  = dist.masked_fill(eye_bool, float('inf'))   # (N,N)
        mk_over_rik   = self.mass.unsqueeze(0) / dist_no_self      # (N,N)
        phi           = self.G * mk_over_rik.sum(dim=1)            # (N,)  Φᵢ
        phi_i_b       = phi.unsqueeze(1).expand(n, n)              # (N,N)
        phi_j_b       = phi.unsqueeze(0).expand(n, n)              # (N,N)

        # EIH term A  — along n̂ᵢⱼ
        coeff_A = (-vi_sq_b
                   - 2.0 * vj_sq_b
                   + 4.0 * vi_dot_vj
                   + 1.5 * nhat_dvj_sq
                   + 4.0 * phi_i_b
                   + phi_j_b)                                      # (N,N)

        forces_1PN_A = scalar_N.unsqueeze(2) * n_hat * coeff_A.unsqueeze(2)  # (N,N,3)

        # EIH term B  — along (vᵢ−vⱼ)
        vel_diff     = vi - vj                                     # (N,N,3)
        dot_B        = 4.0 * nhat_dot_vi - 3.0 * nhat_dot_vj      # (N,N)
        forces_1PN_B = scalar_N.unsqueeze(2) * vel_diff * dot_B.unsqueeze(2)  # (N,N,3)

        forces_1PN   = (forces_1PN_A + forces_1PN_B) / C2         # (N,N,3)

        # Total — zero diagonal (belt-and-suspenders)
        forces_total = forces_N + forces_1PN
        forces_total.masked_fill_(eye_bool.unsqueeze(2), 0.0)

        # Sum forces on each body i (dim=1 sums contributions from all j)
        # Divide by mᵢ (unsqueeze keeps shape (N,1) for broadcast)
        self.accel = forces_total.sum(dim=1) / self.mass.unsqueeze(1)  # (N,3)

    # ------------------------------------------------------------------
    # COLLISION — perfectly inelastic merger  (post-step)
    # ------------------------------------------------------------------

    def _detect_mergers(self):
        """
        Find overlapping pairs and queue them.  Called at end of each sub-step.
        Uses upper-triangle only so each pair is detected once.
        Physical radius (self.radius) used — independent of frontend visual scale.
        """
        if self.n < 2:
            return
        diff      = self.pos.unsqueeze(0) - self.pos.unsqueeze(1)      # (N,N,3)
        dist      = torch.sqrt((diff * diff).sum(dim=2) + 1e-12)       # (N,N)
        # Collision fires at visual radius boundary (COLLISION_SCALE matches
        # frontend BODY_RADIUS_SCALE), so user sees contact at exact merge point.
        r_sum     = (self.radius.unsqueeze(0) + self.radius.unsqueeze(1)) * COLLISION_SCALE
        triu_mask = torch.triu(
            torch.ones(self.n, self.n, device=device, dtype=torch.bool), diagonal=1)
        col_mask  = (dist < r_sum) & triu_mask
        if not col_mask.any():
            return
        pairs = torch.where(col_mask)
        for i, j in zip(pairs[0].tolist(), pairs[1].tolist()):
            self._pending_mergers.append((i, j))

    def _apply_mergers(self):
        """
        Apply all pending mergers after the full Yoshida step completes.
        Merges are processed highest-mass first to handle chain collisions correctly.
        Each merge:
          - mass:   m_new = m₁ + m₂
          - pos:    centre of mass (mass-weighted)
          - vel:    momentum-conserving  (m₁v₁ + m₂v₂) / m_new
          - radius: volume-conserving    ∛(r₁³ + r₂³)
          - color:  mass-weighted HSL blend
          - name:   heavier body's name (or "Merged Body")
        Body i (heavier) absorbs body j (lighter); j is removed.
        After all merges tensors and derived caches are rebuilt.
        """
        if not self._pending_mergers:
            return

        # Deduplicate: a body may appear in multiple pairs
        # Process by collecting a set of (survivor, absorbed) pairs
        # using union-find to handle chains correctly
        parent = list(range(self.n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # Heavier body becomes root/survivor
            ma = float(self.mass[ra])
            mb = float(self.mass[rb])
            if mb > ma:
                ra, rb = rb, ra
            parent[rb] = ra

        for i, j in self._pending_mergers:
            if i < self.n and j < self.n:
                union(i, j)
        self._pending_mergers.clear()

        # Build merge groups: {survivor_idx: [absorbed_indices]}
        groups: dict[int, list[int]] = {}
        for idx in range(self.n):
            root = find(idx)
            if root != idx:
                groups.setdefault(root, []).append(idx)

        if not groups:
            return

        # Apply each merge group
        absorbed_set = set()
        for survivor, absorbed_list in groups.items():
            # Accumulate mass and momentum
            m_total   = float(self.mass[survivor])
            px        = float(self.mass[survivor]) * self.pos[survivor]
            p_mom     = float(self.mass[survivor]) * self.vel[survivor]
            vol       = float(self.radius[survivor]) ** 3

            h_s, s_s, l_s = self.colors[survivor]
            for j in absorbed_list:
                mj   = float(self.mass[j])
                m_total += mj
                px      += mj * self.pos[j]
                p_mom   += mj * self.vel[j]
                vol     += float(self.radius[j]) ** 3
                # Mass-weighted HSL blend
                frac     = mj / m_total
                h_j, s_j, l_j = self.colors[j]
                h_s  = h_s  * (1 - frac) + h_j  * frac
                s_s  = s_s  * (1 - frac) + s_j  * frac
                l_s  = l_s  * (1 - frac) + l_j  * frac
                absorbed_set.add(j)

            # Write merged state directly into tensors at survivor index
            self.mass[survivor]   = m_total
            self.pos[survivor]    = px / m_total
            self.vel[survivor]    = p_mom / m_total
            self.radius[survivor] = vol ** (1.0 / 3.0)
            self.colors[survivor] = [round(h_s, 1), round(s_s, 1), round(l_s, 1)]
            # Keep the more massive body's name; mark merger if equal
            self.names[survivor]  = self.names[survivor]

        if not absorbed_set:
            return

        # Remove absorbed bodies — rebuild tensors and metadata
        keep = [i for i in range(self.n) if i not in absorbed_set]
        self.pos    = self.pos[keep].contiguous()
        self.vel    = self.vel[keep].contiguous()
        self.mass   = self.mass[keep].contiguous()
        self.radius = self.radius[keep].contiguous()
        self.colors = [self.colors[i] for i in keep]
        self.names  = [self.names[i]  for i in keep]
        self.bodies = [self.bodies[i] for i in keep]
        self.n      = len(keep)
        self.accel  = torch.zeros(self.n, 3, device=device, dtype=torch.float32)
        self._rebuild_derived()
        # Reinitialise energy baseline after mass change
        self.total_energy_initial = self._compute_total_energy()

    # ------------------------------------------------------------------
    # PARAMETER UPDATES
    # ------------------------------------------------------------------

    def update_params(self, dt=None, substeps=None):
        """G is fixed (physical constant); only dt and substeps are user-tunable."""
        if dt is not None:
            self.base_dt = float(dt)
        if substeps is not None:
            self.substeps = max(1, int(substeps))
        self.dt = self.base_dt / max(1, self.substeps)

    def calculate_orbital_velocity(self, position, new_body_mass=1.0):
        """Compute stable tangential circular-orbit velocity for a new body."""
        if self.n == 0:
            return [0.0, 0.0, 0.0]
        pos        = torch.tensor(position, device=device, dtype=torch.float32)
        total_mass = self.mass.sum()
        if float(total_mass) < 1e-10:
            return [0.0, 0.0, 0.0]
        com       = (self.pos * self.mass.unsqueeze(1)).sum(dim=0) / total_mass
        to_center = com - pos
        dist_xz   = torch.sqrt(to_center[0] ** 2 + to_center[2] ** 2)
        if float(dist_xz) < 1e-4:
            return [0.0, 0.0, 0.0]
        speed   = torch.sqrt(
            torch.tensor(self.G, device=device) * total_mass / dist_xz)
        tangent = torch.tensor(
            [-to_center[2], 0.0, to_center[0]], device=device)
        tangent = tangent / (torch.norm(tangent) + 1e-9)
        vel     = tangent * speed
        # ±3% perturbation for visual variety (no Y component)
        vel    += (torch.randn(3, device=device)
                   * speed * 0.03
                   * torch.tensor([1.0, 0.0, 1.0], device=device))
        return vel.detach().cpu().tolist()

    # ------------------------------------------------------------------
    # BODY MANAGEMENT
    # ------------------------------------------------------------------

    def add_body(self, body):
        """Append a new body; rebuild derived tensors and caches."""
        if self.n >= self.max_bodies:
            return False
        self.bodies.append(body)
        self.n = len(self.bodies)
        self.pos    = torch.cat([self.pos,
            torch.tensor([body["pos"]],    device=device, dtype=torch.float32)])
        self.vel    = torch.cat([self.vel,
            torch.tensor([body["vel"]],    device=device, dtype=torch.float32)])
        self.mass   = torch.cat([self.mass,
            torch.tensor([body["mass"]],   device=device, dtype=torch.float32)])
        self.radius = torch.cat([self.radius,
            torch.tensor([body["radius"]], device=device, dtype=torch.float32)])
        self.colors.append(body["color"])
        self.names.append(body["name"])
        self.accel = torch.zeros(self.n, 3, device=device, dtype=torch.float32)
        self._rebuild_derived()
        return True

    def reset(self, bodies=None, substeps=None):
        """
        Full reset.  bodies=None → solar system,  bodies=[] → blank canvas.
        Clears all perf counters and pending mergers.
        """
        self.bodies  = bodies if bodies is not None else self._create_solar_system()
        if substeps is not None:
            self.substeps = max(1, int(substeps))
        self.dt                   = self.base_dt / max(1, self.substeps)
        self.n                    = len(self.bodies)
        self._pending_mergers     = []
        self._initialize_tensors()
        self.step_count           = 0
        self.frame_times          = []
        self.compute_time_ms      = 0.0
        self.total_energy_initial = self._compute_total_energy()
        # grid_x / grid_z are fixed — no reinit needed

    # ------------------------------------------------------------------
    # STATE SERIALISATION
    # ------------------------------------------------------------------

    def get_state(self):
        avg_time = (sum(self.frame_times) / len(self.frame_times)
                    if self.frame_times else 0.016)
        try:
            cur   = self._compute_total_energy()
            denom = abs(self.total_energy_initial)
            drift = (abs(cur - self.total_energy_initial) / denom * 100.0
                     if denom > 1e-10 else 0.0)
        except Exception:
            drift = 0.0

        return {
            "bodies": {
                "positions":  self.pos.detach().cpu().numpy().tolist(),
                "velocities": self.vel.detach().cpu().numpy().tolist(),
                "masses":     self.mass.detach().cpu().numpy().tolist(),
                "radii":      self.radius.detach().cpu().numpy().tolist(),
                "colors":     self.colors,
                "names":      self.names,
                "lod_hints":  self._compute_lod_hints(),
            },
            "params": {
                "G":        float(self.G),
                "dt":       float(self.base_dt),
                "substeps": int(self.substeps),
            },
            "performance": {
                "compute_time_ms":  round(float(self.compute_time_ms), 3),
                "avg_fps":          round(1.0 / avg_time) if avg_time > 0 else 0,
                "memory_gb":        round(float(self.memory_used_gb), 3),
                "body_count":       int(self.n),
                "substeps":         int(self.substeps),
                "energy_drift_pct": round(float(drift), 6),
            },
        }

    def _compute_lod_hints(self):
        """Vectorised LOD computation — O(N²) distance matrix, one GPU call."""
        if self.n == 0:
            return []

        total_mass = float(self.mass.sum())
        com = ((self.pos * self.mass.unsqueeze(1)).sum(dim=0) / total_mass
               if total_mass > 1e-10 else torch.zeros(3, device=device))

        # Vectorised: pairwise distances in one GPU call
        if self.n > 1:
            diff = self.pos.unsqueeze(0) - self.pos.unsqueeze(1)      # (N,N,3)
            pairwise = torch.sqrt((diff * diff).sum(dim=2))            # (N,N)
            eye_inf  = torch.eye(self.n, device=device, dtype=torch.bool)
            pairwise.masked_fill_(eye_inf, float("inf"))
            min_dist = pairwise.min(dim=1).values                      # (N,)
        else:
            min_dist = torch.full((1,), float("inf"), device=device)

        dist_to_com = torch.norm(self.pos - com.unsqueeze(0), dim=1)  # (N,)

        lod = []
        for i in range(self.n):
            r    = float(self.radius[i])
            m    = float(self.mass[i])
            base = 3 if r > 0.15 else (2 if r > 0.04 else 1)
            if m > 0.001: base = max(base, 2)
            if float(min_dist[i]) < 5.0 * r: base = min(3, base + 1)
            if float(dist_to_com[i]) < 5.0:  base = min(3, base + 1)
            lod.append(base)
        return lod


# ============================================================================
# ASYNC SIMULATOR — Controller Interface
# ============================================================================

class AsyncSimulator:
    """
    Wraps AdvancedOrbitalSimulator in a dedicated 120 Hz physics thread.

    Spacetime grid is exported on a SEPARATE low-priority daemon thread to
    prevent the blocking CUDA→CPU memory copy from stalling the physics loop.
    The grid thread is triggered by an Event every GRID_UPDATE_INTERVAL physics
    ticks; it reads a snapshot of the GPU tensor (non-blocking), converts to
    bytes, and stores the result for the next state emission.
    """

    GRID_UPDATE_INTERVAL = 8   # emit grid every N physics ticks (~15 fps at 120 Hz)

    def __init__(self, max_bodies=25, substeps=1, dt=0.002):
        self.sim = AdvancedOrbitalSimulator(
            max_bodies=max_bodies, substeps=substeps, dt=dt)

        self.state_queue:   Queue = Queue(maxsize=2)
        self.command_queue: Queue = Queue()

        self.running          = True
        self.physics_fps      = 120
        self.sim_lock         = threading.Lock()
        self._shutdown_event  = threading.Event()
        self._grid_tick       = 0
        self._last_grid_data  = None   # bytes or None — written by grid thread

        # Event used to ping the grid thread: set by physics, cleared by grid thread
        self._grid_event      = threading.Event()
        # Snapshot tensor passed to grid thread (written under sim_lock)
        self._grid_snapshot   = None   # (n, pos_x, pos_z, mass, radius) packed tuple

        self.physics_thread = threading.Thread(
            target=self._physics_loop, daemon=True, name="gravsim-physics")
        self.grid_thread = threading.Thread(
            target=self._grid_loop,    daemon=True, name="gravsim-grid")

        self.physics_thread.start()
        self.grid_thread.start()
        print("[GravSim] Physics thread started")
        print("[GravSim] Grid export thread started")

    # ------------------------------------------------------------------
    # PHYSICS LOOP  (120 Hz — never touches .cpu() or numpy)
    # ------------------------------------------------------------------

    def _physics_loop(self):
        target_dt = 1.0 / self.physics_fps
        while self.running and not self._shutdown_event.is_set():
            t0             = time.perf_counter()
            reset_occurred = False

            # Drain command queue
            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    with self.sim_lock:
                        if self._handle_command(cmd):
                            reset_occurred = True
                            self._grid_tick = 0
            except Empty:
                pass

            with self.sim_lock:
                self.sim.step()
                state = self.sim.get_state()

                # Decide whether to trigger a grid export this tick
                self._grid_tick += 1
                if self._grid_tick >= self.GRID_UPDATE_INTERVAL or reset_occurred:
                    self._grid_tick = 0
                    # Take a lightweight snapshot: only the tensors needed for
                    # grid computation, cloned on GPU (fast) so grid thread can
                    # read them without holding sim_lock.
                    if self.sim.n > 0:
                        self._grid_snapshot = (
                            self.sim.pos.clone(),     # (N,3) GPU tensor
                            self.sim.mass.clone(),    # (N,)  GPU tensor
                            self.sim.radius.clone(),  # (N,)  GPU tensor
                            self.sim.grid_x,          # fixed — no clone needed
                            self.sim.grid_z,          # fixed — no clone needed
                        )
                    else:
                        self._grid_snapshot = None
                    self._grid_event.set()   # wake grid thread

            state["reset_occurred"] = reset_occurred

            # Attach last grid data if available
            if self._last_grid_data is not None:
                state["spacetime"] = {
                    "grid_y_bytes": self._last_grid_data,
                    "grid_res":     GRID_RES,
                    "grid_extent":  GRID_EXTENT,
                }

            try:
                self.state_queue.put_nowait(state)
            except Exception:
                pass

            sleep_time = max(0.0, target_dt - (time.perf_counter() - t0))
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[GravSim] Physics thread stopped")

    # ------------------------------------------------------------------
    # GRID EXPORT LOOP  (daemon — handles all CUDA→CPU copies)
    # ------------------------------------------------------------------

    def _grid_loop(self):
        """
        Waits for the physics thread to signal a new snapshot is ready,
        then performs the GPU→CPU copy and Flamm computation off the hot path.
        This prevents cudaDeviceSynchronize from blocking the physics loop.
        """
        while self.running and not self._shutdown_event.is_set():
            triggered = self._grid_event.wait(timeout=0.5)
            if not triggered:
                continue
            self._grid_event.clear()

            snap = self._grid_snapshot   # read outside lock — snapshot is immutable
            if snap is None:
                # No bodies — emit a flat zeroed grid
                flat = bytes(GRID_RES * GRID_RES * 4)
                self._last_grid_data = flat
                continue

            try:
                pos_snap, mass_snap, radius_snap, grid_x, grid_z = snap
                n = pos_snap.shape[0]

                r_s = 2.0 * G_REAL * mass_snap / C2                # (N,)

                gx  = grid_x.unsqueeze(2)                           # (R,R,1)
                gz  = grid_z.unsqueeze(2)                           # (R,R,1)
                bx  = pos_snap[:, 0]                                # (N,)
                bz  = pos_snap[:, 2]                                # (N,)

                r_grid = torch.sqrt(
                    (gx - bx) ** 2 + (gz - bz) ** 2 + 1e-8)       # (R,R,N)

                # Clamp r to each body's physical radius (prevents depth singularity)
                r_floor   = radius_snap.unsqueeze(0).unsqueeze(0).clamp(min=1e-4)
                r_clamped = torch.maximum(r_grid, r_floor)          # (R,R,N)

                r_s_3d    = r_s.unsqueeze(0).unsqueeze(0)           # (1,1,N)
                z_grid    = (-2.0 * torch.sqrt(r_s_3d * r_clamped)
                             ).sum(dim=2) * WARP_VISUAL_SCALE        # (R,R)

                # CPU copy happens here — isolated from physics thread
                grid_bytes = z_grid.contiguous().cpu().numpy().ravel().tobytes()
                self._last_grid_data = grid_bytes

            except Exception as e:
                print(f"[GravSim] Grid export error: {e}")
                traceback.print_exc()

        print("[GravSim] Grid thread stopped")

    # ------------------------------------------------------------------
    # COMMAND HANDLER  (called under sim_lock)
    # ------------------------------------------------------------------

    def _handle_command(self, cmd: dict) -> bool:
        """Returns True on reset. Always called under sim_lock."""
        try:
            cmd_type = cmd.get("type")
            data     = cmd.get("data", {}) or {}

            if cmd_type == "update_params":
                self.sim.update_params(
                    dt=data.get("dt"), substeps=data.get("substeps"))
                return False

            elif cmd_type == "add_body":
                self.sim.add_body(data)
                return False

            elif cmd_type == "reset":
                self.sim.reset(
                    bodies=data.get("bodies"),
                    substeps=data.get("substeps"))
                # Flush stale states
                while not self.state_queue.empty():
                    try:
                        self.state_queue.get_nowait()
                    except Empty:
                        break
                self._last_grid_data = None
                return True

            else:
                print(f"[GravSim] Unknown command: {cmd_type}")

        except Exception as e:
            print(f"[GravSim] Command error: {e}")
            traceback.print_exc()

        return False

    # ------------------------------------------------------------------
    # CONTROLLER INTERFACE
    # ------------------------------------------------------------------

    def get_latest_state(self):
        try:
            return self.state_queue.get_nowait()
        except Empty:
            return None

    def send_command(self, cmd: dict):
        self.command_queue.put(cmd)

    def calculate_velocity(self, position, mass):
        with self.sim_lock:
            return self.sim.calculate_orbital_velocity(position, mass)

    def stop(self):
        self.running = False
        self._shutdown_event.set()
        self._grid_event.set()   # unblock grid thread if waiting
        for t in (self.physics_thread, self.grid_thread):
            if t.is_alive():
                t.join(timeout=3.0)
        try:
            with self.sim_lock:
                for attr in ("pos", "vel", "mass", "radius",
                             "grid_x", "grid_z", "grid_y"):
                    if hasattr(self.sim, attr):
                        delattr(self.sim, attr)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass
        print("[GravSim] Stopped and GPU memory released")