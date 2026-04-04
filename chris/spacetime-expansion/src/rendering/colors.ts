/**
 * Redshift color mapping.
 *
 * As the universe expands, photons are stretched to longer wavelengths.
 * This is cosmological redshift: z = 1/a - 1.
 *
 * We map z to a visual color shift: blue/white → yellow → orange → red.
 */

/**
 * Apply cosmological redshift to a galaxy's base color.
 *
 * Parameters
 * ----------
 * base : [r, g, b] in 0..255
 * z    : cosmological redshift (0 = present, higher = more redshifted)
 *
 * Returns shifted [r, g, b].
 */
export function redshiftColor(
  base: [number, number, number],
  z: number
): [number, number, number] {
  // Normalize z to a 0..1 "redness" factor
  // At z=0 → factor=0 (original color), z=3+ → factor~1 (deep red)
  const factor = Math.min(z / 3, 1);

  const r = Math.round(base[0] + (255 - base[0]) * factor);
  const g = Math.round(base[1] * (1 - factor * 0.7));
  const b = Math.round(base[2] * (1 - factor));

  return [
    Math.min(255, Math.max(0, r)),
    Math.min(255, Math.max(0, g)),
    Math.min(255, Math.max(0, b)),
  ];
}

/** Convert [r,g,b] to a CSS rgba string. */
export function rgbString(
  color: [number, number, number],
  alpha = 1
): string {
  return `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
}

/** Epoch indicator colors for the timeline bar. */
export const EPOCH_GRADIENT: Record<string, string> = {
  Inflation: '#ffd700',
  Radiation: '#ff6b35',
  Matter: '#4a9eff',
  'Dark Energy': '#c084fc',
};
