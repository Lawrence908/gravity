/**
 * Generate initial galaxy positions in comoving coordinates.
 *
 * In 2D mode: galaxies are distributed uniformly within a circle (not a
 * square — circular distribution avoids the visible square boundary during
 * expansion).
 *
 * In 3D mode: galaxies are distributed uniformly within a sphere.
 *
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
  const { is3D } = config;
  const R = config.universeSize * 0.5; // radius

  const galaxies: Galaxy[] = [];
  const clusters: Cluster[] = [];

  /** Sample a uniformly random point within the universe volume. */
  function randPoint(radiusFraction = 1.0): [number, number, number] {
    if (is3D) {
      // Uniform spherical: r ∝ cbrt(uniform) ensures equal density per shell
      const r = Math.cbrt(rng()) * R * radiusFraction;
      const theta = rng() * 2 * Math.PI;
      const phi = Math.acos(2 * rng() - 1);
      return [
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi),
      ];
    } else {
      // Uniform circular: r ∝ sqrt(uniform) ensures equal density per ring
      const r = Math.sqrt(rng()) * R * radiusFraction;
      const theta = rng() * 2 * Math.PI;
      return [r * Math.cos(theta), r * Math.sin(theta), 0];
    }
  }

  // Place cluster centers first, within 70% of the universe radius so there
  // is room for cluster members to spread around them.
  for (let c = 0; c < config.clusterCount; c++) {
    const [cx, cy, cz] = randPoint(0.7);
    clusters.push({
      id: c,
      centerX: cx,
      centerY: cy,
      centerZ: cz,
      bindingRadius: config.clusterBindingRadius,
      galaxyIds: [],
    });
  }

  // Generate galaxies
  for (let i = 0; i < config.galaxyCount; i++) {
    const [comX, comY, comZ] = randPoint();
    const color = GALAXY_COLORS[Math.floor(rng() * GALAXY_COLORS.length)];

    const galaxy: Galaxy = {
      id: i,
      comovingX: comX,
      comovingY: comY,
      comovingZ: comZ,
      physicalX: comX * A_INITIAL,
      physicalY: comY * A_INITIAL,
      physicalZ: comZ * A_INITIAL,
      clusterId: null,
      clusterOffsetX: 0,
      clusterOffsetY: 0,
      clusterOffsetZ: 0,
      baseColor: color,
    };

    // Check if this galaxy falls within a cluster's binding radius
    for (const cluster of clusters) {
      const dx = comX - cluster.centerX;
      const dy = comY - cluster.centerY;
      const dz = comZ - cluster.centerZ;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < cluster.bindingRadius && cluster.galaxyIds.length < 5) {
        galaxy.clusterId = cluster.id;
        // Physical-space offset from cluster center at initialization.
        // This offset stays constant (gravitational binding).
        galaxy.clusterOffsetX = dx * A_INITIAL;
        galaxy.clusterOffsetY = dy * A_INITIAL;
        galaxy.clusterOffsetZ = dz * A_INITIAL;
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
        dist: Math.sqrt(
          (g.comovingX - cluster.centerX) ** 2 +
          (g.comovingY - cluster.centerY) ** 2 +
          (g.comovingZ - cluster.centerZ) ** 2
        ),
      }))
      .sort((a, b) => a.dist - b.dist);

    for (let j = 0; j < Math.min(needed, unbound.length); j++) {
      const g = unbound[j].g;
      const dx = g.comovingX - cluster.centerX;
      const dy = g.comovingY - cluster.centerY;
      const dz = g.comovingZ - cluster.centerZ;

      g.clusterId = cluster.id;
      // Move the galaxy's comoving position near the cluster center
      g.comovingX = cluster.centerX + dx * 0.3;
      g.comovingY = cluster.centerY + dy * 0.3;
      g.comovingZ = cluster.centerZ + dz * 0.3;
      g.clusterOffsetX = (g.comovingX - cluster.centerX) * A_INITIAL;
      g.clusterOffsetY = (g.comovingY - cluster.centerY) * A_INITIAL;
      g.clusterOffsetZ = (g.comovingZ - cluster.centerZ) * A_INITIAL;
      g.physicalX = g.comovingX * A_INITIAL;
      g.physicalY = g.comovingY * A_INITIAL;
      g.physicalZ = g.comovingZ * A_INITIAL;
      cluster.galaxyIds.push(g.id);
    }
  }

  return { galaxies, clusters };
}
