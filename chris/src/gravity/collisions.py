"""Collision detection and inelastic merging for the gravity simulation.

Optional post-step: merge particle–star and particle–particle pairs when
separation < r_collide. Mass and momentum conserved; energy not (inelastic).
Star is always particle index 0.
"""

from __future__ import annotations

import numpy as np

from .state import ParticleState


def resolve_collisions(
    state: ParticleState,
    r_collide: float,
    star_index: int = 0,
) -> ParticleState:
    """Resolve collisions by inelastic merge; return state with possibly smaller N.

    (1) Star–particle: any particle i with |r_i - r_star| < r_collide is merged
        into the star (star mass += m_i, particle removed). Star position and
        velocity unchanged (toy: star dominates).
    (2) Particle–particle: among remaining particles, repeatedly find the pair
        (i, j) with smallest separation below r_collide; merge into one (COM
        position, momentum-conserving velocity), remove one, until no such pairs.

    Particle index 0 is always the star after resolution. Arrays are compacted
    so indices 1..N-1 are disk/cloud particles.

    Parameters
    ----------
    state : ParticleState
        Current positions, velocities, masses (N, ndim).
    r_collide : float
        Collision radius; pairs with separation < r_collide are merged.
    star_index : int
        Index of the central star (default 0).

    Returns
    -------
    ParticleState
        New state with N' <= N; star at index 0.
    """
    pos = state.positions
    vel = state.velocities
    masses = state.masses
    n = pos.shape[0]
    ndim = pos.shape[1]
    r2_collide = r_collide * r_collide

    if n <= 1:
        return state

    # Work with copies so we can mutate and compact
    p = np.array(pos, dtype=float, copy=True)
    v = np.array(vel, dtype=float, copy=True)
    m = np.array(masses, dtype=float, copy=True)

    # Mask: True = keep this particle
    keep = np.ones(n, dtype=bool)

    # --- (1) Star–particle merges ---
    star_pos = p[star_index]
    star_vel = v[star_index]
    star_mass = m[star_index]

    for i in range(n):
        if i == star_index or not keep[i]:
            continue
        dr = p[i] - star_pos
        r2 = float(np.sum(dr * dr))
        if r2 < r2_collide:
            star_mass += m[i]
            keep[i] = False

    p[star_index] = star_pos
    v[star_index] = star_vel
    m[star_index] = star_mass

    # Compact: keep only star + particles still present (star first, then rest in order)
    indices = np.where(keep)[0]
    others = [i for i in indices if i != star_index]
    order = np.array([star_index] + sorted(others), dtype=np.intp)
    p = p[order]
    v = v[order]
    m = m[order]
    n_curr = len(order)

    if n_curr <= 1:
        return ParticleState(positions=p, velocities=v, masses=m)

    # --- (2) Particle–particle merges (indices 1..n_curr-1 only) ---
    while True:
        # Pairwise distances among non-star particles (indices 1..n_curr-1)
        non_star = p[1:]
        n_ns = non_star.shape[0]
        if n_ns <= 1:
            break
        dx = non_star[:, None, :] - non_star[None, :, :]
        r2 = np.sum(dx * dx, axis=2)
        np.fill_diagonal(r2, np.inf)
        i_min, j_min = np.unravel_index(np.argmin(r2), r2.shape)
        r2_min = r2[i_min, j_min]
        if r2_min >= r2_collide:
            break
        # i_min, j_min are in 1..n_curr-1 space; actual indices are i_min+1, j_min+1
        ii = i_min + 1
        jj = j_min + 1
        mi = m[ii]
        mj = m[jj]
        mtot = mi + mj
        # Merge ii and jj into ii (COM position, momentum-conserving velocity); drop jj
        com = (mi * p[ii] + mj * p[jj]) / mtot
        v_com = (mi * v[ii] + mj * v[jj]) / mtot
        p[ii] = com
        v[ii] = v_com
        m[ii] = mtot
        # Remove jj: compact by taking all rows except jj
        mask = np.ones(n_curr, dtype=bool)
        mask[jj] = False
        p = p[mask]
        v = v[mask]
        m = m[mask]
        n_curr = p.shape[0]

    return ParticleState(positions=p, velocities=v, masses=m)
