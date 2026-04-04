/**
 * Generate initial galaxy positions in comoving coordinates.
 *
 * Galaxies are distributed roughly uniformly across a comoving square.
 * A few clusters of 3-5 galaxies are designated as gravitationally bound.
 */

import type { Galaxy, Cluster, SimulationConfig } from './types';
import { A_INITIAL } from '../physics/constants';

/** Seeded pseudo-random number generator (simple LCG) for reproducibility. */
function makeRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x80000000;
  };
}

/** Random base colors — blue-white palette for galaxies. */
const GALAXY_COLORS: [number, number, number][] = [
  [200, 220, 255], // blue-white
  [180, 200, 255], // pale blue
  [220, 230, 255], // near-white
  [170, 190, 240], // soft blue
  [240, 240, 255], // white
  [160, 180, 230], // steel blue
];

export function generateGalaxies(
  config: SimulationConfig,
  seed = 42
): { galaxies: Galaxy[]; clusters: Cluster[] } {
  const rng = makeRng(seed);

  const galaxies: Galaxy[] = [];
  const clusters: Cluster[] = [];

  // Place cluster centers first, spread across the comoving region
  for (let c = 0; c < config.clusterCount; c++) {
    clusters.push({
      id: c,
      centerX: (rng() - 0.5) * config.universeSize * 0.7,
      centerY: (rng() - 0.5) * config.universeSize * 0.7,
      bindingRadius: config.clusterBindingRadius,
      galaxyIds: [],
    });
  }

  // Generate galaxies
  for (let i = 0; i < config.galaxyCount; i++) {
    const comX = (rng() - 0.5) * config.universeSize;
    const comY = (rng() - 0.5) * config.universeSize;
    const color = GALAXY_COLORS[Math.floor(rng() * GALAXY_COLORS.length)];

    const galaxy: Galaxy = {
      id: i,
      comovingX: comX,
      comovingY: comY,
      physicalX: comX * A_INITIAL,
      physicalY: comY * A_INITIAL,
      clusterId: null,
      clusterOffsetX: 0,
      clusterOffsetY: 0,
      baseColor: color,
    };

    // Check if this galaxy falls within a cluster's binding radius
    for (const cluster of clusters) {
      const dx = comX - cluster.centerX;
      const dy = comY - cluster.centerY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < cluster.bindingRadius && cluster.galaxyIds.length < 5) {
        galaxy.clusterId = cluster.id;
        // Physical-space offset from cluster center at initialization.
        // This offset stays constant (gravitational binding).
        galaxy.clusterOffsetX = dx * A_INITIAL;
        galaxy.clusterOffsetY = dy * A_INITIAL;
        cluster.galaxyIds.push(i);
        break;
      }
    }

    galaxies.push(galaxy);
  }

  // Ensure each cluster has at least 3 galaxies by forcibly assigning
  // nearby unbound galaxies.
  for (const cluster of clusters) {
    if (cluster.galaxyIds.length >= 3) continue;

    const needed = 3 - cluster.galaxyIds.length;
    const unbound = galaxies
      .filter((g) => g.clusterId === null)
      .map((g) => ({
        g,
        dist: Math.hypot(
          g.comovingX - cluster.centerX,
          g.comovingY - cluster.centerY
        ),
      }))
      .sort((a, b) => a.dist - b.dist);

    for (let j = 0; j < Math.min(needed, unbound.length); j++) {
      const g = unbound[j].g;
      const dx = g.comovingX - cluster.centerX;
      const dy = g.comovingY - cluster.centerY;

      g.clusterId = cluster.id;
      // Move the galaxy's comoving position near the cluster center
      // so it visually starts inside the cluster
      g.comovingX = cluster.centerX + dx * 0.3;
      g.comovingY = cluster.centerY + dy * 0.3;
      // Physical-space offset from cluster center at initialization.
      // This offset stays constant (gravitational binding).
      g.clusterOffsetX = (g.comovingX - cluster.centerX) * A_INITIAL;
      g.clusterOffsetY = (g.comovingY - cluster.centerY) * A_INITIAL;
      g.physicalX = g.comovingX * A_INITIAL;
      g.physicalY = g.comovingY * A_INITIAL;
      cluster.galaxyIds.push(g.id);
    }
  }

  return { galaxies, clusters };
}
