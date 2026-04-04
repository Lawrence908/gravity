/**
 * Friedmann equation solver and scale factor integrator.
 *
 * The first Friedmann equation:
 *   H²(a) = H0² × [ Ω_r/a⁴ + Ω_m/a³ + Ω_k/a² + Ω_Λ ]
 *
 * The scale factor evolves as:
 *   da/dt = a × H(a)
 *
 * We integrate with RK4 for accuracy across the enormous dynamic range
 * of a(t) from inflation through dark energy domination.
 */

import {
  H0,
  OMEGA_R,
  OMEGA_M,
  OMEGA_LAMBDA,
  OMEGA_K,
  INFLATION_H,
  INFLATION_END_A,
  DARK_ENERGY_DOMINANCE_A,
  A_INITIAL,
  type Epoch,
} from './constants';

/**
 * Compute the Hubble parameter H(a) from the Friedmann equation.
 *
 * During the inflation phase (a < INFLATION_END_A), we use a constant
 * de Sitter-like Hubble rate for visual clarity.
 */
export function computeHubble(a: number): number {
  if (a < INFLATION_END_A) {
    return INFLATION_H;
  }
  const a2 = a * a;
  const a3 = a2 * a;
  const a4 = a3 * a;
  const H2 =
    H0 * H0 * (OMEGA_R / a4 + OMEGA_M / a3 + OMEGA_K / a2 + OMEGA_LAMBDA);
  return Math.sqrt(Math.max(H2, 0));
}

/**
 * da/dt = a × H(a).  This is the ODE we integrate.
 */
function dadt(a: number): number {
  return a * computeHubble(a);
}

/**
 * Advance the scale factor by dt using 4th-order Runge-Kutta.
 *
 * Includes adaptive sub-stepping: if the relative change |da/a| would
 * exceed 1% in a single step, the step is subdivided.
 */
export function stepScaleFactor(a: number, dt: number): number {
  const MAX_RELATIVE_CHANGE = 0.01;
  let remaining = dt;
  let current = a;

  while (remaining > 1e-15) {
    // Estimate step size needed to keep |da/a| < threshold
    const H = computeHubble(current);
    const maxDt = MAX_RELATIVE_CHANGE / Math.max(H, 1e-10);
    const stepDt = Math.min(remaining, maxDt);

    // RK4 integration
    const k1 = stepDt * dadt(current);
    const k2 = stepDt * dadt(current + k1 / 2);
    const k3 = stepDt * dadt(current + k2 / 2);
    const k4 = stepDt * dadt(current + k3);
    current += (k1 + 2 * k2 + 2 * k3 + k4) / 6;

    remaining -= stepDt;
  }

  return Math.max(current, A_INITIAL);
}

/**
 * Classify the current cosmological epoch based on which energy
 * component dominates the expansion.
 */
export function getEpoch(a: number): Epoch {
  if (a < INFLATION_END_A) return 'Inflation';

  const a3 = a * a * a;
  const a4 = a3 * a;

  const rhoR = OMEGA_R / a4;
  const rhoM = OMEGA_M / a3;
  const rhoL = OMEGA_LAMBDA;

  if (rhoR > rhoM && rhoR > rhoL) return 'Radiation';
  if (rhoM > rhoL) return 'Matter';
  return 'Dark Energy';
}

/**
 * Compute the deceleration parameter q(t) = -a·ä / ȧ².
 * Positive q = decelerating; negative q = accelerating.
 */
export function computeDeceleration(a: number): number {
  if (a < INFLATION_END_A) return -1; // de Sitter: q = -1

  const a3 = a * a * a;
  const a4 = a3 * a;
  const H = computeHubble(a);
  if (H < 1e-15) return 0;

  // From the acceleration equation:
  // q = (Ω_r/a⁴ + Ω_m/(2a³) - Ω_Λ) × H0² / H²
  const q =
    (H0 * H0 * (2 * OMEGA_R / a4 + OMEGA_M / a3 - 2 * OMEGA_LAMBDA)) /
    (2 * H * H);
  return q;
}

/**
 * Check whether dark energy currently dominates (for UI hints).
 */
export function isDarkEnergyDominated(a: number): boolean {
  return a >= DARK_ENERGY_DOMINANCE_A;
}
