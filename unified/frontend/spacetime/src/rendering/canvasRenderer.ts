/**
 * Canvas 2D renderer for the spacetime expansion simulation.
 *
 * Draws: background, comoving grid, galaxies with glow + redshift,
 * velocity vectors, light travel lines, and the HUD overlay.
 *
 * Supports both 2D (orthographic) and 3D (perspective projection with
 * depth sorting) rendering modes.
 */

import type { Galaxy, LightLine } from '../simulation/types';
import type { UniverseState } from '../physics/universe';
import { redshiftColor, rgbString } from './colors';
import {
  EPOCH_COLORS,
  OPACITY_THRESHOLD_A,
  A_INITIAL,
  type Epoch,
} from '../physics/constants';

export interface ViewTransform {
  /**
   * Comoving coordinates of the view centre.
   * In 2D: controls pan.  In 3D: the Hubble-flow reference origin —
   * right-click-drag shifts this so you can stand on any galaxy and see
   * that expansion looks the same from everywhere.
   */
  centerX: number;
  centerY: number;
  centerZ: number;
  /** Pixels per physical-space unit. */
  scale: number;
  /** 3D rotation angles in radians (ignored in 2D mode). */
  rotX: number;
  rotY: number;
}

// ─── 3D Projection ─────────────────────────────────────────

const FOV = 700; // perspective focal length in pixels

interface Projected {
  sx: number;
  sy: number;
  /** Size multiplier from perspective divide (1 = no change). */
  sizeFactor: number;
  /** Rotated Z depth for back-to-front sorting. */
  depth: number;
}

/**
 * Project a 3D physical-space point to canvas screen coordinates.
 * Applies Y-axis rotation (rotY) then X-axis rotation (rotX),
 * then perspective divide.
 *
 * Returns null if the point is behind the camera.
 */
function project3D(
  px: number,
  py: number,
  pz: number,
  rotX: number,
  rotY: number,
  scale: number,
  w: number,
  h: number
): Projected | null {
  // Rotate around Y axis (horizontal yaw)
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
  const x1 = px * cosY + pz * sinY;
  const z1 = -px * sinY + pz * cosY;

  // Rotate around X axis (vertical pitch)
  const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const y2 = py * cosX - z1 * sinX;
  const z2 = py * sinX + z1 * cosX;
  const x2 = x1;

  // Perspective divide: depth = FOV + rotated_z * scale
  const depth = FOV + z2 * scale;
  if (depth < 10) return null; // behind camera

  const perspScale = FOV / depth;
  return {
    sx: w / 2 + x2 * scale * perspScale,
    sy: h / 2 + y2 * scale * perspScale,
    sizeFactor: perspScale,
    depth: z2, // used for sorting (higher = further back)
  };
}

// ─── Background ────────────────────────────────────────────

export function drawBackground(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number
): void {
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, w, h);

  const grad = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w * 0.7);
  grad.addColorStop(0, 'rgba(15, 15, 35, 0)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0.4)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
}

// ─── Comoving Grid (2D only) ──────────────────────────────


/** Convert comoving coords to screen pixel coords. */
function toScreen(
  comX: number,
  comY: number,
  a: number,
  view: ViewTransform,
  canvasW: number,
  canvasH: number
): [number, number] {
  const physX = (comX - view.centerX) * a;
  const physY = (comY - view.centerY) * a;
  return [
    canvasW / 2 + physX * view.scale,
    canvasH / 2 + physY * view.scale,
  ];
}

export function drawComovingGrid(
  ctx: CanvasRenderingContext2D,
  a: number,
  gridSpacing: number,
  view: ViewTransform,
  w: number,
  h: number
): void {
  ctx.strokeStyle = 'rgba(26, 32, 64, 0.6)';
  ctx.lineWidth = 0.5;

  const MAX_LINES = 60;
  const halfW = w / (2 * view.scale * a);
  const halfH = h / (2 * view.scale * a);

  let spacing = gridSpacing;
  const estLines = ((2 * halfW) / spacing + (2 * halfH) / spacing);
  if (estLines > MAX_LINES) {
    spacing = spacing * Math.ceil(estLines / MAX_LINES);
  }

  const minX = Math.floor((view.centerX - halfW) / spacing) * spacing;
  const maxX = Math.ceil((view.centerX + halfW) / spacing) * spacing;
  const minY = Math.floor((view.centerY - halfH) / spacing) * spacing;
  const maxY = Math.ceil((view.centerY + halfH) / spacing) * spacing;

  ctx.beginPath();

  for (let x = minX; x <= maxX; x += spacing) {
    const [sx, sy1] = toScreen(x, minY, a, view, w, h);
    const [, sy2] = toScreen(x, maxY, a, view, w, h);
    ctx.moveTo(sx, sy1);
    ctx.lineTo(sx, sy2);
  }

  for (let y = minY; y <= maxY; y += spacing) {
    const [sx1, sy] = toScreen(minX, y, a, view, w, h);
    const [sx2] = toScreen(maxX, y, a, view, w, h);
    ctx.moveTo(sx1, sy);
    ctx.lineTo(sx2, sy);
  }

  ctx.stroke();
}

// ─── Comoving 3D Grid ──────────────────────────────────────

/**
 * Draw a comoving 3D grid that expands with the scale factor.
 *
 * Renders:
 *  - A flat comoving grid in the XY plane (z = 0)
 *  - Four vertical Z-edges at the bounding-box corners
 *  - Top/bottom bounding-box outlines
 *  - Colored X (red) / Y (green) / Z (blue) axis arrows from the origin
 *
 * All grid positions are in comoving space; physical positions
 * = comoving × a(t), so the grid visibly expands as a(t) grows.
 */
export function draw3DGrid(
  ctx: CanvasRenderingContext2D,
  a: number,
  gridSpacing: number,
  universeSize: number,
  view: ViewTransform,
  w: number,
  h: number
): void {
  const halfSize = universeSize / 2;

  // Cap grid line count for performance and clarity
  const MAX_LINES = 8;
  let spacing = gridSpacing;
  const estLines = Math.ceil((2 * halfSize) / spacing);
  if (estLines > MAX_LINES) {
    spacing *= Math.ceil(estLines / MAX_LINES);
  }

  /** Project comoving coordinates to screen, offset by the view centre. */
  function proj(cx: number, cy: number, cz: number) {
    return project3D(
      (cx - view.centerX) * a,
      (cy - view.centerY) * a,
      (cz - view.centerZ) * a,
      view.rotX, view.rotY, view.scale, w, h,
    );
  }

  /** Draw a line between two comoving 3D points. */
  function drawLine(
    cx1: number, cy1: number, cz1: number,
    cx2: number, cy2: number, cz2: number,
  ) {
    const p1 = proj(cx1, cy1, cz1);
    const p2 = proj(cx2, cy2, cz2);
    if (!p1 || !p2) return;
    ctx.beginPath();
    ctx.moveTo(p1.sx, p1.sy);
    ctx.lineTo(p2.sx, p2.sy);
    ctx.stroke();
  }

  // Build the comoving grid positions for each axis
  const range: number[] = [];
  for (let v = -halfSize; v <= halfSize + 0.001; v += spacing) {
    range.push(v);
  }

  // ── Flat comoving grid in the XY plane (z = 0) ──
  ctx.strokeStyle = 'rgba(26, 42, 90, 0.55)';
  ctx.lineWidth = 0.5;
  for (const x of range) {
    drawLine(x, -halfSize, 0, x, halfSize, 0);
  }
  for (const y of range) {
    drawLine(-halfSize, y, 0, halfSize, y, 0);
  }

  // ── Four vertical Z-edges at bounding-box corners ──
  ctx.strokeStyle = 'rgba(26, 42, 90, 0.30)';
  ctx.lineWidth = 0.4;
  for (const x of [-halfSize, halfSize]) {
    for (const y of [-halfSize, halfSize]) {
      drawLine(x, y, -halfSize, x, y, halfSize);
    }
  }

  // ── Top and bottom bounding-box outlines ──
  ctx.strokeStyle = 'rgba(26, 42, 90, 0.18)';
  ctx.lineWidth = 0.3;
  for (const z of [-halfSize, halfSize]) {
    drawLine(-halfSize, -halfSize, z,  halfSize, -halfSize, z);
    drawLine( halfSize, -halfSize, z,  halfSize,  halfSize, z);
    drawLine( halfSize,  halfSize, z, -halfSize,  halfSize, z);
    drawLine(-halfSize,  halfSize, z, -halfSize, -halfSize, z);
  }

  // ── Colored axis arrows from the view centre ──
  // proj(view.centerX, …) ≡ project3D(0,0,0,…) ≡ screen centre, always (w/2, h/2).
  const axisLen      = halfSize * 1.18;
  const axisOriginSx = w / 2;
  const axisOriginSy = h / 2;

  function drawAxis(
    cx: number, cy: number, cz: number,
    color: string,
    label: string,
  ) {
    const end = proj(cx, cy, cz);
    if (!end) return;

    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    ctx.moveTo(axisOriginSx, axisOriginSy);
    ctx.lineTo(end.sx, end.sy);
    ctx.stroke();

    // Arrowhead
    const dx    = end.sx - axisOriginSx;
    const dy    = end.sy - axisOriginSy;
    const angle = Math.atan2(dy, dx);
    const head  = 8;
    ctx.beginPath();
    ctx.moveTo(end.sx, end.sy);
    ctx.lineTo(end.sx - head * Math.cos(angle - 0.4), end.sy - head * Math.sin(angle - 0.4));
    ctx.moveTo(end.sx, end.sy);
    ctx.lineTo(end.sx - head * Math.cos(angle + 0.4), end.sy - head * Math.sin(angle + 0.4));
    ctx.stroke();

    // Label
    ctx.fillStyle    = color;
    ctx.font         = 'bold 11px "SF Mono", "Fira Code", monospace';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, end.sx + 6, end.sy);
  }

  // Endpoint = view centre + axisLen along each comoving axis
  drawAxis(view.centerX + axisLen, view.centerY,           view.centerZ,           'rgba(255, 85,  85,  0.88)', 'X');
  drawAxis(view.centerX,           view.centerY + axisLen, view.centerZ,           'rgba(85,  220, 85,  0.88)', 'Y');
  drawAxis(view.centerX,           view.centerY,           view.centerZ + axisLen, 'rgba(100, 160, 255, 0.88)', 'Z');

  ctx.textBaseline = 'top'; // restore default
}

// ─── Galaxies ──────────────────────────────────────────────

/** Convert physical coords relative to view center to screen (2D). */
function physToScreen(
  physX: number,
  physY: number,
  view: ViewTransform,
  canvasW: number,
  canvasH: number
): [number, number] {
  return [
    canvasW / 2 + physX * view.scale,
    canvasH / 2 + physY * view.scale,
  ];
}

export function drawGalaxies(
  ctx: CanvasRenderingContext2D,
  galaxies: Galaxy[],
  z: number,
  a: number,
  view: ViewTransform,
  is3D: boolean,
  w: number,
  h: number,
  selectedId: number | null
): void {
  // Only draw glows for smaller counts — gradient per-galaxy is expensive
  const showGlow = galaxies.length <= 250;

  if (is3D) {
    // --- 3D mode: project all, sort back-to-front, draw ---
    type Entry = { galaxy: Galaxy; sx: number; sy: number; sizeFactor: number };
    const visible: Entry[] = [];

    for (const galaxy of galaxies) {
      const proj = project3D(
        galaxy.physicalX - view.centerX * a,
        galaxy.physicalY - view.centerY * a,
        galaxy.physicalZ - view.centerZ * a,
        view.rotX,
        view.rotY,
        view.scale,
        w,
        h
      );
      if (!proj) continue;
      if (proj.sx < -50 || proj.sx > w + 50 || proj.sy < -50 || proj.sy > h + 50) continue;
      visible.push({ galaxy, sx: proj.sx, sy: proj.sy, sizeFactor: proj.sizeFactor });
    }

    // Sort back to front (largest depth first = furthest away)
    visible.sort((a, b) => b.sizeFactor - a.sizeFactor);

    for (const { galaxy, sx, sy, sizeFactor } of visible) {
      const color = redshiftColor(galaxy.baseColor, Math.max(0, z));
      const isSelected = galaxy.id === selectedId;
      const baseRadius = Math.max(1.5, (isSelected ? 5 : 3.5) * sizeFactor);
      const alpha = Math.max(0.3, Math.min(0.95, sizeFactor * 1.2));

      if (showGlow && baseRadius > 2) {
        const glowRadius = baseRadius * 3;
        const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, glowRadius);
        glow.addColorStop(0, rgbString(color, isSelected ? 0.4 * alpha : 0.25 * alpha));
        glow.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(sx, sy, glowRadius, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = rgbString(color, alpha);
      ctx.beginPath();
      ctx.arc(sx, sy, baseRadius, 0, Math.PI * 2);
      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = rgbString(color, 0.6);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(sx, sy, baseRadius + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  } else {
    // --- 2D mode: orthographic projection ---
    for (const galaxy of galaxies) {
      const physRelX = galaxy.physicalX - view.centerX * a;
      const physRelY = galaxy.physicalY - view.centerY * a;
      const [sx, sy] = physToScreen(physRelX, physRelY, view, w, h);

      if (sx < -30 || sx > w + 30 || sy < -30 || sy > h + 30) continue;

      const color = redshiftColor(galaxy.baseColor, Math.max(0, z));
      const isSelected = galaxy.id === selectedId;
      const baseRadius = isSelected ? 5 : 3.5;

      if (showGlow) {
        const glowRadius = baseRadius * 3;
        const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, glowRadius);
        glow.addColorStop(0, rgbString(color, isSelected ? 0.5 : 0.3));
        glow.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(sx, sy, glowRadius, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = rgbString(color, 0.95);
      ctx.beginPath();
      ctx.arc(sx, sy, baseRadius, 0, Math.PI * 2);
      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = rgbString(color, 0.6);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(sx, sy, baseRadius + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
}

// ─── Velocity Vectors (2D + 3D) ───────────────────────────

/**
 * Draw Hubble-flow velocity arrows for each galaxy.
 *
 * In 2D mode the arrows are computed and drawn directly in screen space.
 * In 3D mode the galaxy position and a small step along the velocity
 * direction are both projected through the perspective transform so the
 * arrows foreshorten correctly with depth.
 */
export function drawVelocityVectors(
  ctx: CanvasRenderingContext2D,
  galaxies: Galaxy[],
  H: number,
  a: number,
  view: ViewTransform,
  is3D: boolean,
  w: number,
  h: number
): void {
  const arrowLen = 25;
  const headLen  = 4;

  if (is3D) {
    // Velocity arrows are relative to the view centre (the observer's position).
    // Shifting the centre to another galaxy shows that expansion looks the same
    // from everywhere — there is no preferred centre.
    const cx = view.centerX * a;
    const cy = view.centerY * a;
    const cz = view.centerZ * a;

    for (const galaxy of galaxies) {
      const relX = galaxy.physicalX - cx;
      const relY = galaxy.physicalY - cy;
      const relZ = galaxy.physicalZ - cz;
      const d = Math.sqrt(relX * relX + relY * relY + relZ * relZ);
      if (d < 0.01) continue;

      const p0 = project3D(relX, relY, relZ, view.rotX, view.rotY, view.scale, w, h);
      if (!p0) continue;
      if (p0.sx < -50 || p0.sx > w + 50 || p0.sy < -50 || p0.sy > h + 50) continue;

      // Project a step along the recession direction for perspective-correct direction.
      const step  = d * 0.25;
      const p1    = project3D(
        relX + (relX / d) * step,
        relY + (relY / d) * step,
        relZ + (relZ / d) * step,
        view.rotX, view.rotY, view.scale, w, h
      );
      if (!p1) continue;

      const scrDx  = p1.sx - p0.sx;
      const scrDy  = p1.sy - p0.sy;
      const scrLen = Math.sqrt(scrDx * scrDx + scrDy * scrDy);
      if (scrLen < 0.1) continue;

      const speed    = H * d;
      const normLen  = Math.min(speed * p0.sizeFactor * view.scale * 0.5, arrowLen);
      if (normLen < 2) continue;

      const ndx   = scrDx / scrLen;
      const ndy   = scrDy / scrLen;
      const ex    = p0.sx + ndx * normLen;
      const ey    = p0.sy + ndy * normLen;
      const angle = Math.atan2(ndy, ndx);

      ctx.strokeStyle = 'rgba(100, 220, 130, 0.40)';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.moveTo(p0.sx, p0.sy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - headLen * Math.cos(angle - 0.4), ey - headLen * Math.sin(angle - 0.4));
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - headLen * Math.cos(angle + 0.4), ey - headLen * Math.sin(angle + 0.4));
      ctx.stroke();
    }
    return;
  }

  // ── 2D mode ──
  for (const galaxy of galaxies) {
    const physRelX = galaxy.physicalX - view.centerX * a;
    const physRelY = galaxy.physicalY - view.centerY * a;
    const [sx, sy] = physToScreen(physRelX, physRelY, view, w, h);

    if (sx < -30 || sx > w + 30 || sy < -30 || sy > h + 30) continue;

    const d = Math.sqrt(physRelX * physRelX + physRelY * physRelY);
    if (d < 0.01) continue;

    const speed   = H * d;
    const normLen = Math.min(speed * view.scale * 0.5, arrowLen);
    if (normLen < 2) continue;

    const dirX  = physRelX / d;
    const dirY  = physRelY / d;
    const endX  = sx + dirX * normLen;
    const endY  = sy + dirY * normLen;
    const angle = Math.atan2(dirY, dirX);

    ctx.strokeStyle = 'rgba(100, 220, 130, 0.35)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - headLen * Math.cos(angle - 0.4), endY - headLen * Math.sin(angle - 0.4));
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - headLen * Math.cos(angle + 0.4), endY - headLen * Math.sin(angle + 0.4));
    ctx.stroke();
  }
}

// ─── Light Travel Lines ────────────────────────────────────

export function drawLightLines(
  ctx: CanvasRenderingContext2D,
  lines: LightLine[],
  galaxies: Galaxy[],
  a: number,
  view: ViewTransform,
  is3D: boolean,
  w: number,
  h: number
): void {
  for (const line of lines) {
    const g1 = galaxies[line.fromId];
    const g2 = galaxies[line.toId];
    if (!g1 || !g2) continue;

    const alpha = Math.max(0, 0.25 * (1 - line.age));
    let sx1: number, sy1: number, sx2: number, sy2: number;

    if (is3D) {
      const cx = view.centerX * a, cy = view.centerY * a, cz = view.centerZ * a;
      const p1 = project3D(g1.physicalX - cx, g1.physicalY - cy, g1.physicalZ - cz, view.rotX, view.rotY, view.scale, w, h);
      const p2 = project3D(g2.physicalX - cx, g2.physicalY - cy, g2.physicalZ - cz, view.rotX, view.rotY, view.scale, w, h);
      if (!p1 || !p2) continue;
      sx1 = p1.sx; sy1 = p1.sy; sx2 = p2.sx; sy2 = p2.sy;
    } else {
      const physRelX1 = g1.physicalX - view.centerX * a;
      const physRelY1 = g1.physicalY - view.centerY * a;
      const physRelX2 = g2.physicalX - view.centerX * a;
      const physRelY2 = g2.physicalY - view.centerY * a;
      [sx1, sy1] = physToScreen(physRelX1, physRelY1, view, w, h);
      [sx2, sy2] = physToScreen(physRelX2, physRelY2, view, w, h);
    }

    const grad = ctx.createLinearGradient(sx1, sy1, sx2, sy2);
    grad.addColorStop(0, `rgba(100, 150, 255, ${alpha})`);
    grad.addColorStop(1, `rgba(255, 100, 80, ${alpha * 0.7})`);

    ctx.strokeStyle = grad;
    ctx.lineWidth = 0.8;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(sx1, sy1);
    ctx.lineTo(sx2, sy2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// ─── Opacity Fog (early universe) ──────────────────────────

/**
 * Draws a warm radial fog overlay during the photon–baryon plasma era,
 * when the universe was opaque to light.  The fog fades out as a(t)
 * approaches OPACITY_THRESHOLD_A (recombination).
 */
export function drawOpacityFog(
  ctx: CanvasRenderingContext2D,
  scaleFactor: number,
  w: number,
  h: number
): void {
  if (scaleFactor >= OPACITY_THRESHOLD_A) return;

  // t = 1 at A_INITIAL (densest plasma), t = 0 at recombination threshold
  const t = (OPACITY_THRESHOLD_A - scaleFactor) / (OPACITY_THRESHOLD_A - A_INITIAL);
  const fogAlpha = t * 0.48;

  // Warm radial glow — hot ionised plasma scattering all photons
  const grad = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.max(w, h) * 0.8);
  grad.addColorStop(0,   `rgba(255, 160, 60, ${fogAlpha * 0.8})`);
  grad.addColorStop(0.5, `rgba(210, 70,  20, ${fogAlpha * 0.55})`);
  grad.addColorStop(1,   `rgba(160, 20,  5,  ${fogAlpha * 0.25})`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Text banner — only when fog is strong enough to read
  if (t > 0.18) {
    const textAlpha = Math.min(1, (t - 0.18) / 0.35);
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = `rgba(255, 220, 160, ${textAlpha})`;
    ctx.font = 'bold 12px system-ui, sans-serif';
    ctx.fillText('Photon–baryon plasma  ·  Universe opaque to light', w / 2, h - 38);
    ctx.fillStyle = `rgba(255, 195, 130, ${textAlpha * 0.85})`;
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(
      'Temperature too high for neutral atoms  ·  CMB photons not yet free to travel',
      w / 2,
      h - 22
    );
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.restore();
  }
}

// ─── Epoch Legend + Colour Scale ───────────────────────────

/** One-line description of the expansion regime for each epoch. */
const EPOCH_EXPANSION_DESC: Record<string, string> = {
  Inflation:     '↑↑ Exponential (de Sitter)',
  Radiation:     '↓  Decelerating (photons)',
  Matter:        '↓  Decelerating (gravity)',
  'Dark Energy': '↑  Accelerating (Λ)',
};

/**
 * Draws a legend panel (bottom-left) listing all four cosmological epochs
 * with their expansion behaviours, plus a redshift colour scale explaining
 * what the galaxy colours represent.
 *
 * The active epoch is highlighted; all others are dimmed.
 */
export function drawEpochLegend(
  ctx: CanvasRenderingContext2D,
  state: UniverseState,
  _w: number,
  h: number
): void {
  const panelW = 284;
  const panelH = 208;
  const margin = 16;
  const px = margin;
  const py = h - panelH - 50;  // 50px above hint text + bottom controls bar

  // Panel background
  ctx.fillStyle = 'rgba(10, 10, 26, 0.82)';
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(px, py, panelW, panelH, 6);
  } else {
    ctx.rect(px, py, panelW, panelH);
  }
  ctx.fill();

  const lx = px + 12;
  let ly = py + 12;

  ctx.textBaseline = 'top';

  // Section title
  ctx.fillStyle = 'rgba(156, 163, 175, 0.65)';
  ctx.font = '10px "SF Mono", "Fira Code", monospace';
  ctx.fillText('COSMOLOGICAL EPOCHS', lx, ly);
  ly += 15;

  const ORDERED_EPOCHS: Epoch[] = ['Inflation', 'Radiation', 'Matter', 'Dark Energy'];

  for (const epoch of ORDERED_EPOCHS) {
    const isActive = epoch === state.epoch;
    const color = EPOCH_COLORS[epoch];

    // Colored dot
    ctx.globalAlpha = isActive ? 1 : 0.45;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lx + 5, ly + 7, isActive ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;

    // Epoch name
    ctx.fillStyle = isActive ? '#ffffff' : 'rgba(195, 200, 210, 0.5)';
    ctx.font = isActive
      ? 'bold 12px system-ui, sans-serif'
      : '12px system-ui, sans-serif';
    ctx.fillText(epoch, lx + 16, ly);

    // Expansion description on the second line, inset
    ctx.fillStyle = isActive
      ? 'rgba(228, 228, 232, 0.85)'
      : 'rgba(115, 125, 140, 0.5)';
    ctx.font = '10px "SF Mono", "Fira Code", monospace';
    ctx.fillText(EPOCH_EXPANSION_DESC[epoch], lx + 16, ly + 14);

    ly += isActive ? 36 : 30;
  }

  // Divider
  ctx.strokeStyle = 'rgba(228, 228, 232, 0.12)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(px + 8, ly + 2);
  ctx.lineTo(px + panelW - 8, ly + 2);
  ctx.stroke();
  ly += 12;

  // Colour scale section title
  ctx.fillStyle = 'rgba(156, 163, 175, 0.65)';
  ctx.font = '10px "SF Mono", "Fira Code", monospace';
  ctx.fillText('GALAXY COLOR = COSMOLOGICAL REDSHIFT', lx, ly);
  ly += 13;

  // Gradient bar: blue-white → yellow → orange → deep red
  const barX = lx;
  const barW = panelW - 26;
  const barH = 7;

  const colorGrad = ctx.createLinearGradient(barX, ly, barX + barW, ly);
  colorGrad.addColorStop(0,    'rgba(180, 215, 255, 0.92)');  // blue-white  z ≈ 0
  colorGrad.addColorStop(0.33, 'rgba(255, 205, 80,  0.92)');  // yellow      z ≈ 1
  colorGrad.addColorStop(0.66, 'rgba(255, 120, 40,  0.92)');  // orange      z ≈ 2
  colorGrad.addColorStop(1,    'rgba(215, 35,  20,  0.92)');  // deep red    z ≥ 3
  ctx.fillStyle = colorGrad;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(barX, ly, barW, barH, 3);
  } else {
    ctx.rect(barX, ly, barW, barH);
  }
  ctx.fill();
  ly += barH + 5;

  ctx.font = '10px system-ui, sans-serif';
  ctx.fillStyle = 'rgba(156, 163, 175, 0.75)';
  ctx.textAlign = 'left';
  ctx.fillText('z = 0  (now)', barX, ly);
  ctx.textAlign = 'right';
  ctx.fillText('z = 3+  (distant past)', barX + barW, ly);
  ctx.textAlign = 'left';
}

// ─── HUD ───────────────────────────────────────────────────

export function drawHUD(
  ctx: CanvasRenderingContext2D,
  state: UniverseState,
  is3D: boolean,
  w: number,
  h: number
): void {
  const x = 16;
  let y = 24;
  const lineH = 20;

  ctx.font = '13px "SF Mono", "Fira Code", "Cascadia Code", monospace';
  ctx.textBaseline = 'top';

  ctx.fillStyle = 'rgba(10, 10, 26, 0.75)';
  const panelW = 260;
  const panelH = 152;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x - 8, y - 8, panelW, panelH, 6);
  } else {
    const r = 6;
    const px = x - 8, py = y - 8;
    ctx.moveTo(px + r, py);
    ctx.arcTo(px + panelW, py, px + panelW, py + panelH, r);
    ctx.arcTo(px + panelW, py + panelH, px, py + panelH, r);
    ctx.arcTo(px, py + panelH, px, py, r);
    ctx.arcTo(px, py, px + panelW, py, r);
    ctx.closePath();
  }
  ctx.fill();

  const epochColor = EPOCH_COLORS[state.epoch] ?? '#ffffff';
  ctx.fillStyle = epochColor;
  ctx.fillText(`● ${state.epoch}`, x, y);
  y += lineH + 4;

  ctx.fillStyle = '#e4e4e8';
  ctx.fillText(`a(t)  = ${state.scaleFactor.toFixed(4)}`, x, y);
  y += lineH;
  ctx.fillText(`H(t)  = ${state.hubble.toFixed(4)}`, x, y);
  y += lineH;

  ctx.fillStyle = '#9ca3af';
  ctx.fillText(`z     = ${state.redshift.toFixed(2)}`, x, y);
  y += lineH;
  ctx.fillText(`time  = ${state.time.toFixed(2)}`, x, y);
  y += lineH;

  // Expansion behaviour for the current epoch, tinted in the epoch colour
  ctx.fillStyle = epochColor;
  ctx.fillText(EPOCH_EXPANSION_DESC[state.epoch] ?? '', x, y);

  // Instructions hint
  ctx.fillStyle = 'rgba(156, 163, 175, 0.5)';
  ctx.font = '11px system-ui, sans-serif';
  const hint = is3D
    ? 'Left-drag: rotate  |  Right-drag: shift reference  |  Scroll: zoom'
    : 'Click a galaxy to recenter  |  Scroll to zoom  |  Drag to pan';
  ctx.fillText(hint, x, h - 16);

  // Title
  ctx.fillStyle = 'rgba(228, 228, 232, 0.7)';
  ctx.font = '14px system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Cosmological Spacetime Expansion', w / 2, 16);
  ctx.textAlign = 'left';
}
