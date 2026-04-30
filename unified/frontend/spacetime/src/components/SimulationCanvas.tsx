/**
 * SimulationCanvas — the core React component.
 *
 * Owns the requestAnimationFrame loop, canvas ref, view transform
 * (pan/zoom/recenter in 2D; rotate/zoom/pan in 3D), and orchestrates
 * physics stepping + rendering.
 *
 * Performance: simulation state lives in useRef (not useState) to avoid
 * re-renders every frame.  Only the timeline bar uses useState (~3 Hz).
 */

import { useRef, useEffect, useCallback, useState } from 'react';
import { createInitialUniverse, stepUniverse } from '../physics/universe';
import type { UniverseState } from '../physics/universe';
import { generateGalaxies } from '../simulation/galaxyGenerator';
import {
  updateGalaxyPositions,
  updateLightLines,
} from '../simulation/simulationEngine';
import type { Galaxy, Cluster, LightLine } from '../simulation/types';
import { DEFAULT_CONFIG } from '../simulation/types';
import type { ViewTransform } from '../rendering/canvasRenderer';
import {
  drawBackground,
  drawComovingGrid,
  draw3DGrid,
  drawGalaxies,
  drawVelocityVectors,
  drawLightLines,
  drawHUD,
  drawOpacityFog,
  drawEpochLegend,
} from '../rendering/canvasRenderer';
import type { SimConfig } from '../App';

export interface SimulationOptions {
  playing: boolean;
  speed: number;
  showGrid: boolean;
  showVectors: boolean;
  showLightLines: boolean;
}

interface Props {
  options: SimulationOptions;
  simConfig: SimConfig;
  onStateUpdate: (state: UniverseState) => void;
  resetSignal: number;
}

function makeRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x80000000;
  };
}

export default function SimulationCanvas({
  options,
  simConfig,
  onStateUpdate,
  resetSignal,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef   = useRef<CanvasRenderingContext2D | null>(null);
  const rafRef   = useRef<number>(0);

  const universeRef   = useRef<UniverseState>(createInitialUniverse());
  const galaxiesRef   = useRef<Galaxy[]>([]);
  const clustersRef   = useRef<Cluster[]>([]);
  const lightLinesRef = useRef<LightLine[]>([]);
  const rngRef        = useRef(makeRng(123));

  const viewRef = useRef<ViewTransform>({
    centerX: 0,
    centerY: 0,
    centerZ: 0,
    scale: 40,
    rotX: 0.25,
    rotY: 0,
  });
  const selectedGalaxyRef = useRef<number | null>(null);

  const isDragging    = useRef(false);
  const dragButton    = useRef<number>(0);
  const dragStart     = useRef({ x: 0, y: 0 });
  const dragViewStart = useRef({ x: 0, y: 0, z: 0, rotX: 0, rotY: 0 });

  const optionsRef     = useRef(options);
  optionsRef.current   = options;
  const simConfigRef   = useRef(simConfig);
  simConfigRef.current = simConfig;

  const lastHudUpdate = useRef(0);

  // ─── Timeline / history ──────────────────────────────────────

  const historyRef     = useRef<UniverseState[]>([]);
  const lastHistTime   = useRef(-Infinity);
  const isScrubbingRef = useRef(false);

  const [timeline, setTimeline] = useState({ len: 0, pos: 0 });

  // ─── Initialize / reset ──────────────────────────────────────

  const initialize = useCallback(() => {
    const seed = Math.floor(Math.random() * 100000);
    const cfg  = { ...DEFAULT_CONFIG, ...simConfigRef.current };
    const { galaxies, clusters } = generateGalaxies(cfg, seed);
    galaxiesRef.current   = galaxies;
    clustersRef.current   = clusters;
    lightLinesRef.current = [];
    universeRef.current   = createInitialUniverse(cfg.omegaLambda);
    rngRef.current        = makeRng(seed + 1);
    selectedGalaxyRef.current = null;

    historyRef.current     = [];
    lastHistTime.current   = -Infinity;
    isScrubbingRef.current = false;
    setTimeline({ len: 0, pos: 0 });

    viewRef.current = {
      centerX: 0,
      centerY: 0,
      centerZ: 0,
      scale: 40,
      rotX: cfg.is3D ? 0.25 : 0,
      rotY: 0,
    };
    onStateUpdate(universeRef.current);
  }, [onStateUpdate]);

  useEffect(() => { initialize(); }, [resetSignal, initialize]);

  // ─── Canvas resize ────────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const dpr  = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width  = rect.width  * dpr;
      canvas.height = rect.height * dpr;
      const ctx = canvas.getContext('2d');
      if (ctx) { ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctxRef.current = ctx; }
    };
    resize();
    const obs = new ResizeObserver(resize);
    obs.observe(canvas);
    return () => obs.disconnect();
  }, []);

  // ─── Animation loop ───────────────────────────────────────────

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let lastTime   = 0;
    let frameCount = 0;

    const animate = (timestamp: number) => {
      rafRef.current = requestAnimationFrame(animate);
      const dt = lastTime === 0 ? 16 : Math.min(timestamp - lastTime, 50);
      lastTime = timestamp;
      frameCount++;

      const ctx = ctxRef.current;
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const w   = canvas.width  / dpr;
      const h   = canvas.height / dpr;

      const opts = optionsRef.current;
      const is3D = simConfigRef.current.is3D;

      if (is3D && opts.playing && !isDragging.current) {
        viewRef.current.rotY += (dt / 1000) * 0.15;
      }

      if (opts.playing && !isScrubbingRef.current) {
        const simDt = (dt / 1000) * opts.speed * 0.5;
        universeRef.current = stepUniverse(universeRef.current, simDt);
        updateGalaxyPositions(galaxiesRef.current, clustersRef.current, universeRef.current.scaleFactor);

        if (opts.showLightLines) {
          lightLinesRef.current = updateLightLines(
            lightLinesRef.current, galaxiesRef.current.length, simDt, rngRef.current,
          );
        }

        if (timestamp - lastHudUpdate.current > 100) {
          lastHudUpdate.current = timestamp;
          onStateUpdate(universeRef.current);
        }

        if (universeRef.current.time - lastHistTime.current >= 0.05) {
          historyRef.current.push({ ...universeRef.current });
          lastHistTime.current = universeRef.current.time;
          if (frameCount % 10 === 0) {
            const n = historyRef.current.length;
            setTimeline({ len: n, pos: n - 1 });
          }
        }
      }

      const universe = universeRef.current;
      const view     = viewRef.current;

      drawBackground(ctx, w, h);
      drawOpacityFog(ctx, universe.scaleFactor, w, h);

      if (opts.showGrid) {
        if (is3D) {
          draw3DGrid(ctx, universe.scaleFactor, DEFAULT_CONFIG.gridSpacing, DEFAULT_CONFIG.universeSize, view, w, h);
        } else {
          drawComovingGrid(ctx, universe.scaleFactor, DEFAULT_CONFIG.gridSpacing, view, w, h);
        }
      }

      if (opts.showLightLines && !isScrubbingRef.current) {
        drawLightLines(ctx, lightLinesRef.current, galaxiesRef.current, universe.scaleFactor, view, is3D, w, h);
      }

      drawGalaxies(ctx, galaxiesRef.current, universe.redshift, universe.scaleFactor, view, is3D, w, h, selectedGalaxyRef.current);

      if (opts.showVectors) {
        drawVelocityVectors(ctx, galaxiesRef.current, universe.hubble, universe.scaleFactor, view, is3D, w, h);
      }

      drawHUD(ctx, universe, is3D, w, h);
      drawEpochLegend(ctx, universe, w, h);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [onStateUpdate]);

  // ─── Timeline handler ─────────────────────────────────────────

  const handleTimeline = useCallback((rawIdx: number) => {
    const hist = historyRef.current;
    if (!hist.length) return;
    const idx     = Math.max(0, Math.min(rawIdx, hist.length - 1));
    const isAtEnd = idx >= hist.length - 1;
    isScrubbingRef.current = !isAtEnd;
    const snap = hist[idx];
    universeRef.current = { ...snap };
    updateGalaxyPositions(galaxiesRef.current, clustersRef.current, snap.scaleFactor);
    setTimeline(t => ({ ...t, pos: idx }));
  }, []);

  // ─── Mouse interaction ─────────────────────────────────────────

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0 && e.button !== 2) return;
    isDragging.current  = true;
    dragButton.current  = e.button;
    dragStart.current   = { x: e.clientX, y: e.clientY };
    dragViewStart.current = {
      x: viewRef.current.centerX, y: viewRef.current.centerY, z: viewRef.current.centerZ,
      rotX: viewRef.current.rotX, rotY: viewRef.current.rotY,
    };
    const canvas = canvasRef.current;
    if (canvas) canvas.style.cursor = e.button === 2 ? 'move' : 'grabbing';
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;

    if (simConfigRef.current.is3D) {
      if (dragButton.current === 0) {
        viewRef.current.rotY = dragViewStart.current.rotY + dx * 0.005;
        viewRef.current.rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, dragViewStart.current.rotX + dy * 0.005));
      } else {
        // Right-drag: pan the reference origin in 3D (camera-space translation).
        // Use rotation at drag-start so direction stays stable while rotating.
        const cosX = Math.cos(dragViewStart.current.rotX);
        const sinX = Math.sin(dragViewStart.current.rotX);
        const cosY = Math.cos(dragViewStart.current.rotY);
        const sinY = Math.sin(dragViewStart.current.rotY);
        const s    = viewRef.current.scale * universeRef.current.scaleFactor;
        // Camera basis rows of R_X·R_Y:
        //   right = (cosY,      0,    sinY       )
        //   down  = (sinX·sinY, cosX, -sinX·cosY )
        viewRef.current.centerX = dragViewStart.current.x - (dx * cosY + dy * sinX * sinY) / s;
        viewRef.current.centerY = dragViewStart.current.y - (dy * cosX)                    / s;
        viewRef.current.centerZ = dragViewStart.current.z - (dx * sinY - dy * sinX * cosY) / s;
      }
    } else {
      const a = universeRef.current.scaleFactor;
      viewRef.current.centerX = dragViewStart.current.x - dx / (viewRef.current.scale * a);
      viewRef.current.centerY = dragViewStart.current.y - dy / (viewRef.current.scale * a);
    }
  }, []);

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    const wasDrag = Math.abs(e.clientX - dragStart.current.x) > 3 || Math.abs(e.clientY - dragStart.current.y) > 3;
    isDragging.current = false;
    const canvas = canvasRef.current;
    if (canvas) canvas.style.cursor = 'grab';
    if (wasDrag || e.button !== 0) return;

    if (simConfigRef.current.is3D || !canvas) return;
    const rect   = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const a      = universeRef.current.scaleFactor;
    const view   = viewRef.current;

    let bestDist = 30;
    let bestId: number | null = null;
    for (const g of galaxiesRef.current) {
      const sx   = rect.width  / 2 + (g.physicalX - view.centerX * a) * view.scale;
      const sy   = rect.height / 2 + (g.physicalY - view.centerY * a) * view.scale;
      const dist = Math.hypot(clickX - sx, clickY - sy);
      if (dist < bestDist) { bestDist = dist; bestId = g.id; }
    }
    if (bestId !== null) {
      const g = galaxiesRef.current[bestId];
      const aVal = universeRef.current.scaleFactor;
      viewRef.current.centerX = aVal > 0 ? g.physicalX / aVal : g.comovingX;
      viewRef.current.centerY = aVal > 0 ? g.physicalY / aVal : g.comovingY;
      selectedGalaxyRef.current = bestId;
    }
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.1 : 0.9;
    viewRef.current.scale = Math.max(0.2, Math.min(5000, viewRef.current.scale * f));
  }, []);

  // ─── Render ────────────────────────────────────────────────────

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
        onContextMenu={(e) => e.preventDefault()}
        style={{ width: '100%', height: '100%', display: 'block', cursor: 'grab' }}
      />

      {timeline.len > 2 && (
        <div className="timeline-bar">
          <span className="timeline-time">
            t&thinsp;=&thinsp;{universeRef.current.time.toFixed(2)}
          </span>

          <input
            type="range"
            className="timeline-slider"
            min={0}
            max={timeline.len - 1}
            value={timeline.pos}
            onChange={(e) => handleTimeline(parseInt(e.target.value))}
          />

          <span
            className={`timeline-live${!isScrubbingRef.current ? ' active' : ''}`}
            onClick={() => handleTimeline(timeline.len - 1)}
            title="Jump to live"
          >
            LIVE
          </span>
        </div>
      )}
    </div>
  );
}
