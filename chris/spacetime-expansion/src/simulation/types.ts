/**
 * Core types for the spacetime expansion simulation.
 */

export interface Galaxy {
  id: number;
  /** Fixed comoving x-coordinate — does not change over time. */
  comovingX: number;
  /** Fixed comoving y-coordinate — does not change over time. */
  comovingY: number;
  /** Physical x = comoving × a(t), recomputed each frame. */
  physicalX: number;
  /** Physical y = comoving × a(t), recomputed each frame. */
  physicalY: number;
  /** If non-null, this galaxy belongs to a gravitationally bound cluster. */
  clusterId: number | null;
  /**
   * For clustered galaxies: fixed physical-space offset from the cluster
   * center.  This offset does NOT scale with a(t), modeling gravitational
   * binding overcoming Hubble expansion.
   */
  clusterOffsetX: number;
  clusterOffsetY: number;
  /** Base color [r, g, b] before redshift is applied. */
  baseColor: [number, number, number];
}

export interface Cluster {
  id: number;
  /** Comoving position of the cluster center. */
  centerX: number;
  centerY: number;
  /** Comoving radius within which galaxies are gravitationally bound. */
  bindingRadius: number;
  /** IDs of galaxies in this cluster. */
  galaxyIds: number[];
}

export interface LightLine {
  fromId: number;
  toId: number;
  /** Normalized age 0..1 — fades out over time. */
  age: number;
}

export interface SimulationConfig {
  galaxyCount: number;
  clusterCount: number;
  clusterBindingRadius: number;
  /** Comoving grid spacing. */
  gridSpacing: number;
  /** Size of the comoving universe region. */
  universeSize: number;
}

export const DEFAULT_CONFIG: SimulationConfig = {
  galaxyCount: 60,
  clusterCount: 4,
  clusterBindingRadius: 0.8,
  gridSpacing: 2.0,
  universeSize: 20,
};
