/**
 * Simulation engine: updates galaxy physical positions each frame.
 *
 * - Unbound galaxies: physicalPos = comovingPos × a(t)  (pure Hubble flow)
 * - Clustered galaxies: clusterCenter × a(t) + fixed physical offset
 *   (gravitational binding overcomes expansion on small scales)
 */

import type { Galaxy, Cluster, LightLine } from './types';

/**
 * Update all galaxy physical positions for the current scale factor.
 */
export function updateGalaxyPositions(
  galaxies: Galaxy[],
  clusters: Cluster[],
  a: number
): void {
  for (const galaxy of galaxies) {
    if (galaxy.clusterId === null) {
      // Unbound galaxy: pure Hubble flow
      galaxy.physicalX = galaxy.comovingX * a;
      galaxy.physicalY = galaxy.comovingY * a;
    } else {
      // Bound galaxy: cluster center expands, but offset stays fixed
      const cluster = clusters[galaxy.clusterId];
      galaxy.physicalX = cluster.centerX * a + galaxy.clusterOffsetX;
      galaxy.physicalY = cluster.centerY * a + galaxy.clusterOffsetY;
    }
  }
}

/**
 * Compute the recession velocity between two galaxies.
 * v_rec = H(t) × d_physical  (Hubble's Law)
 */
export function computeRecessionVelocity(
  g1: Galaxy,
  g2: Galaxy,
  H: number
): { vx: number; vy: number; speed: number } {
  const dx = g2.physicalX - g1.physicalX;
  const dy = g2.physicalY - g1.physicalY;
  const d = Math.sqrt(dx * dx + dy * dy);
  const v = H * d;
  if (d < 1e-10) return { vx: 0, vy: 0, speed: 0 };
  return {
    vx: (dx / d) * v,
    vy: (dy / d) * v,
    speed: v,
  };
}

/**
 * Manage light travel lines — periodically spawn new lines between
 * random galaxy pairs and age out old ones.
 */
export function updateLightLines(
  lines: LightLine[],
  galaxyCount: number,
  dt: number,
  rng: () => number
): LightLine[] {
  // Age existing lines
  const updated = lines
    .map((l) => ({ ...l, age: l.age + dt * 0.3 }))
    .filter((l) => l.age < 1.0);

  // Occasionally spawn new lines (roughly 1 per 30 frames)
  if (rng() < 0.035 && galaxyCount > 1) {
    const fromId = Math.floor(rng() * galaxyCount);
    let toId = Math.floor(rng() * galaxyCount);
    while (toId === fromId) toId = Math.floor(rng() * galaxyCount);
    updated.push({ fromId, toId, age: 0 });
  }

  return updated;
}
