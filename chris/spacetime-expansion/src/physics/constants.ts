/**
 * Cosmological constants and epoch boundaries for the Friedmann model.
 *
 * We use dimensionless simulation units where the present epoch has a(t) = 1.
 * The Hubble parameter H0 is normalized so that the simulation runs at a
 * visually pleasant pace.
 */

/** Present-day Hubble parameter in simulation time units. */
export const H0 = 0.07;

/** Radiation density parameter (Omega_r ~ 9e-5 in real universe). */
export const OMEGA_R = 9e-5;

/** Matter density parameter (baryonic + dark matter). */
export const OMEGA_M = 0.31;

/** Dark energy density parameter (cosmological constant). */
export const OMEGA_LAMBDA = 0.69;

/** Spatial curvature parameter (0 = flat universe — observationally confirmed). */
export const OMEGA_K = 0.0;

/**
 * Inflation parameters.
 * Real inflation spans ~60 e-folds in ~10^-32 s.  We compress this into a
 * visually perceptible phase by using a moderate Hubble rate and a short
 * duration in simulation time.
 */
export const INFLATION_H = 2.0;
export const INFLATION_END_A = 0.005;

/** Scale factor at which dark energy begins to dominate over matter. */
export const DARK_ENERGY_DOMINANCE_A = 0.77;

/** Initial scale factor at the start of the simulation. */
export const A_INITIAL = 0.001;

/** Cosmological epoch names. */
export type Epoch = 'Inflation' | 'Radiation' | 'Matter' | 'Dark Energy';

/** Colors associated with each epoch for HUD display. */
export const EPOCH_COLORS: Record<Epoch, string> = {
  Inflation: '#ffd700',
  Radiation: '#ff6b35',
  Matter: '#4a9eff',
  'Dark Energy': '#c084fc',
};
