/**
 * Canvas 2D renderer for the spacetime expansion simulation.
 *
 * Draws: background, comoving grid, galaxies with glow + redshift,
 * velocity vectors, light travel lines, and the HUD overlay.
 */

import type { Galaxy, LightLine } from '../simulation/types';
import type { UniverseState } from '../physics/universe';
import { redshiftColor, rgbString } from './colors';
import { EPOCH_COLORS } from '../physics/constants';

export interface ViewTransform {
  /** Comoving coordinates of the view center. */
  centerX: number;
  centerY: number;
  /** Pixels per physical-space unit. */
  scale: number;
}

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

/** Convert physical coords (already offset from view center) to screen. */
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

// ─── Background ────────────────────────────────────────────

export function drawBackground(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number
): void {
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, w, h);

  // Subtle radial vignette for depth
  const grad = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w * 0.7);
  grad.addColorStop(0, 'rgba(15, 15, 35, 0)');
  grad.addColorStop(1, 'rgba(0, 0, 0, 0.4)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
}

// ─── Comoving Grid ─────────────────────────────────────────

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

  // Determine grid range in comoving coords visible on screen.
  // When a is very small, the comoving extent is huge — dynamically
  // increase grid spacing to cap the number of lines drawn.
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

  // Vertical lines
  for (let x = minX; x <= maxX; x += spacing) {
    const [sx, sy1] = toScreen(x, minY, a, view, w, h);
    const [, sy2] = toScreen(x, maxY, a, view, w, h);
    ctx.moveTo(sx, sy1);
    ctx.lineTo(sx, sy2);
  }

  // Horizontal lines
  for (let y = minY; y <= maxY; y += spacing) {
    const [sx1, sy] = toScreen(minX, y, a, view, w, h);
    const [sx2] = toScreen(maxX, y, a, view, w, h);
    ctx.moveTo(sx1, sy);
    ctx.lineTo(sx2, sy);
  }

  ctx.stroke();
}

// ─── Galaxies ──────────────────────────────────────────────

export function drawGalaxies(
  ctx: CanvasRenderingContext2D,
  galaxies: Galaxy[],
  z: number,
  a: number,
  view: ViewTransform,
  w: number,
  h: number,
  selectedId: number | null
): void {
  for (const galaxy of galaxies) {
    // Compute screen position from physical coords
    const physRelX = galaxy.physicalX - view.centerX * a;
    const physRelY = galaxy.physicalY - view.centerY * a;
    const [sx, sy] = physToScreen(physRelX, physRelY, view, w, h);

    // Skip if off-screen (with margin)
    if (sx < -30 || sx > w + 30 || sy < -30 || sy > h + 30) continue;

    const color = redshiftColor(galaxy.baseColor, Math.max(0, z));
    const isSelected = galaxy.id === selectedId;
    const baseRadius = isSelected ? 5 : 3.5;

    // Glow layer
    const glowRadius = baseRadius * 3;
    const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, glowRadius);
    glow.addColorStop(0, rgbString(color, isSelected ? 0.5 : 0.3));
    glow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(sx, sy, glowRadius, 0, Math.PI * 2);
    ctx.fill();

    // Core dot
    ctx.fillStyle = rgbString(color, 0.95);
    ctx.beginPath();
    ctx.arc(sx, sy, baseRadius, 0, Math.PI * 2);
    ctx.fill();

    // Selection ring
    if (isSelected) {
      ctx.strokeStyle = rgbString(color, 0.6);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(sx, sy, baseRadius + 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

// ─── Velocity Vectors ──────────────────────────────────────

export function drawVelocityVectors(
  ctx: CanvasRenderingContext2D,
  galaxies: Galaxy[],
  H: number,
  a: number,
  view: ViewTransform,
  w: number,
  h: number
): void {
  const arrowLen = 25; // max arrow length in pixels

  for (const galaxy of galaxies) {
    const physRelX = galaxy.physicalX - view.centerX * a;
    const physRelY = galaxy.physicalY - view.centerY * a;
    const [sx, sy] = physToScreen(physRelX, physRelY, view, w, h);

    if (sx < -30 || sx > w + 30 || sy < -30 || sy > h + 30) continue;

    // Recession velocity direction = away from view center in physical space
    const d = Math.sqrt(physRelX * physRelX + physRelY * physRelY);
    if (d < 0.01) continue;

    const speed = H * d;
    const normLen = Math.min(speed * view.scale * 0.5, arrowLen);
    if (normLen < 2) continue;

    const dirX = physRelX / d;
    const dirY = physRelY / d;
    const endX = sx + dirX * normLen;
    const endY = sy + dirY * normLen;

    ctx.strokeStyle = 'rgba(100, 220, 130, 0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Arrowhead
    const headLen = 4;
    const angle = Math.atan2(dirY, dirX);
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - headLen * Math.cos(angle - 0.4),
      endY - headLen * Math.sin(angle - 0.4)
    );
    ctx.moveTo(endX, endY);
    ctx.lineTo(
      endX - headLen * Math.cos(angle + 0.4),
      endY - headLen * Math.sin(angle + 0.4)
    );
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
  w: number,
  h: number
): void {
  for (const line of lines) {
    const g1 = galaxies[line.fromId];
    const g2 = galaxies[line.toId];
    if (!g1 || !g2) continue;

    const alpha = Math.max(0, 0.25 * (1 - line.age));
    const physRelX1 = g1.physicalX - view.centerX * a;
    const physRelY1 = g1.physicalY - view.centerY * a;
    const physRelX2 = g2.physicalX - view.centerX * a;
    const physRelY2 = g2.physicalY - view.centerY * a;
    const [sx1, sy1] = physToScreen(physRelX1, physRelY1, view, w, h);
    const [sx2, sy2] = physToScreen(physRelX2, physRelY2, view, w, h);

    // Color gradient from blue (emitted) to red (received) to represent redshift
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

// ─── HUD ───────────────────────────────────────────────────

export function drawHUD(
  ctx: CanvasRenderingContext2D,
  state: UniverseState,
  w: number,
  _h: number
): void {
  const x = 16;
  let y = 24;
  const lineH = 20;

  ctx.font = '13px "SF Mono", "Fira Code", "Cascadia Code", monospace';
  ctx.textBaseline = 'top';

  // Background panel
  ctx.fillStyle = 'rgba(10, 10, 26, 0.75)';
  const panelW = 260;
  const panelH = 130;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x - 8, y - 8, panelW, panelH, 6);
  } else {
    // Fallback for browsers without roundRect support
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

  // Epoch label with color indicator
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

  // Instructions (bottom left)
  ctx.fillStyle = 'rgba(156, 163, 175, 0.5)';
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText(
    'Click a galaxy to recenter  |  Scroll to zoom  |  Drag to pan',
    x,
    _h - 16
  );

  // Title
  ctx.fillStyle = 'rgba(228, 228, 232, 0.7)';
  ctx.font = '14px system-ui, -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Cosmological Spacetime Expansion', w / 2, 16);
  ctx.textAlign = 'left';
}
