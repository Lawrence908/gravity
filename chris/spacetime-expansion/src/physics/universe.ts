/**
 * Universe state container and time-stepping.
 */

import type { Epoch } from './constants';
import { A_INITIAL } from './constants';
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
}

/** Create the initial universe state. */
export function createInitialUniverse(): UniverseState {
  const a = A_INITIAL;
  return {
    time: 0,
    scaleFactor: a,
    hubble: computeHubble(a),
    redshift: 1 / a - 1,
    epoch: getEpoch(a),
  };
}

/** Advance the universe by one time step dt. */
export function stepUniverse(
  state: UniverseState,
  dt: number
): UniverseState {
  const newA = stepScaleFactor(state.scaleFactor, dt);
  const newH = computeHubble(newA);
  return {
    time: state.time + dt,
    scaleFactor: newA,
    hubble: newH,
    redshift: 1 / newA - 1,
    epoch: getEpoch(newA),
  };
}
