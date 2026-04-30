/**
 * Universe state container and time-stepping.
 */

import type { Epoch } from './constants';
import { A_INITIAL, OMEGA_LAMBDA } from './constants';
import { computeHubble, stepScaleFactor, getEpoch } from './friedmann';

export interface UniverseState {
  /** Simulation time (arbitrary units). */
  time: number;
  /** Scale factor a(t).  a = 1 corresponds to "present day". */
  scaleFactor: number;
  /** Hubble parameter H(t) = ȧ/a. */
  hubble: number;
  /** Cosmological redshift z = 1/a - 1. */
  redshift: number;
  /** Current dominant epoch. */
  epoch: Epoch;
  /** Dark energy density parameter used for this simulation run. */
  omegaLambda: number;
}

/** Create the initial universe state. */
export function createInitialUniverse(omegaLambda = OMEGA_LAMBDA): UniverseState {
  const a = A_INITIAL;
  return {
    time: 0,
    scaleFactor: a,
    hubble: computeHubble(a, omegaLambda),
    redshift: 1 / a - 1,
    epoch: getEpoch(a, omegaLambda),
    omegaLambda,
  };
}

/** Advance the universe by one time step dt. */
export function stepUniverse(
  state: UniverseState,
  dt: number
): UniverseState {
  const { omegaLambda } = state;
  const newA = stepScaleFactor(state.scaleFactor, dt, omegaLambda);
  const newH = computeHubble(newA, omegaLambda);
  return {
    time: state.time + dt,
    scaleFactor: newA,
    hubble: newH,
    redshift: 1 / newA - 1,
    epoch: getEpoch(newA, omegaLambda),
    omegaLambda,
  };
}
