/**
 * SimulationCanvas — the core React component.
 *
 * Owns the requestAnimationFrame loop, canvas ref, view transform
 * (pan/zoom/recenter), and orchestrates physics stepping + rendering.
 *
 * Performance note: simulation state lives in useRef (not useState) to
 * avoid React re-renders every frame.  HUD values are lifted to the
 * parent via a throttled callback (~10 Hz).
 */

import { useRef, useEffect, useCallback } from 'react';
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
  drawGalaxies,
  drawVelocityVectors,
  drawLightLines,
  drawHUD,
} from '../rendering/canvasRenderer';

export interface SimulationOptions {
  playing: boolean;
  speed: number;
  showGrid: boolean;
  showVectors: boolean;
  showLightLines: boolean;
}

interface Props {
  options: SimulationOptions;
  onStateUpdate: (state: UniverseState) => void;
  resetSignal: number;
}

// Simple RNG for light lines
function makeRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x80000000;
  };
}

export default function SimulationCanvas({
  options,
  onStateUpdate,
  resetSignal,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const rafRef = useRef<number>(0);

  // Simulation state (mutable refs to avoid re-renders)
  const universeRef = useRef<UniverseState>(createInitialUniverse());
  const galaxiesRef = useRef<Galaxy[]>([]);
  const clustersRef = useRef<Cluster[]>([]);
  const lightLinesRef = useRef<LightLine[]>([]);
  const rngRef = useRef(makeRng(123));

  // View state
  const viewRef = useRef<ViewTransform>({
    centerX: 0,
    centerY: 0,
    scale: 40,
  });
  const selectedGalaxyRef = useRef<number | null>(null);

  // Drag state
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const dragViewStart = useRef({ x: 0, y: 0 });

  // Options ref (so the rAF loop reads current values without re-render)
  const optionsRef = useRef(options);
  optionsRef.current = options;

  // Throttled state update for HUD
  const lastHudUpdate = useRef(0);

  // Initialize / reset
  const initialize = useCallback(() => {
    const seed = Math.floor(Math.random() * 100000);
    const { galaxies, clusters } = generateGalaxies(DEFAULT_CONFIG, seed);
    galaxiesRef.current = galaxies;
    clustersRef.current = clusters;
    lightLinesRef.current = [];
    universeRef.current = createInitialUniverse();
    rngRef.current = makeRng(seed + 1);
    selectedGalaxyRef.current = null;
    viewRef.current = { centerX: 0, centerY: 0, scale: 40 };
    onStateUpdate(universeRef.current);
  }, [onStateUpdate]);

  // Reset when resetSignal changes
  useEffect(() => {
    initialize();
  }, [resetSignal, initialize]);

  // Canvas resize handler
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Reset transform before applying DPR scale to prevent
        // cumulative scaling on repeated resizes
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctxRef.current = ctx;
      }
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let lastTime = 0;

    const animate = (timestamp: number) => {
      rafRef.current = requestAnimationFrame(animate);

      const dt = lastTime === 0 ? 16 : Math.min(timestamp - lastTime, 50);
      lastTime = timestamp;

      const ctx = ctxRef.current;
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;

      const opts = optionsRef.current;

      // Step physics
      if (opts.playing) {
        const simDt = (dt / 1000) * opts.speed * 0.5;
        universeRef.current = stepUniverse(universeRef.current, simDt);

        // Update galaxy positions
        updateGalaxyPositions(
          galaxiesRef.current,
          clustersRef.current,
          universeRef.current.scaleFactor
        );

        // Update light lines
        if (opts.showLightLines) {
          lightLinesRef.current = updateLightLines(
            lightLinesRef.current,
            galaxiesRef.current.length,
            simDt,
            rngRef.current
          );
        }

        // Throttled HUD update (~10 Hz)
        if (timestamp - lastHudUpdate.current > 100) {
          lastHudUpdate.current = timestamp;
          onStateUpdate(universeRef.current);
        }
      }

      const universe = universeRef.current;
      const view = viewRef.current;

      // Draw everything
      drawBackground(ctx, w, h);

      if (opts.showGrid) {
        drawComovingGrid(
          ctx,
          universe.scaleFactor,
          DEFAULT_CONFIG.gridSpacing,
          view,
          w,
          h
        );
      }

      if (opts.showLightLines) {
        drawLightLines(
          ctx,
          lightLinesRef.current,
          galaxiesRef.current,
          universe.scaleFactor,
          view,
          w,
          h
        );
      }

      drawGalaxies(
        ctx,
        galaxiesRef.current,
        universe.redshift,
        universe.scaleFactor,
        view,
        w,
        h,
        selectedGalaxyRef.current
      );

      if (opts.showVectors) {
        drawVelocityVectors(
          ctx,
          galaxiesRef.current,
          universe.hubble,
          universe.scaleFactor,
          view,
          w,
          h
        );
      }

      drawHUD(ctx, universe, w, h);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [onStateUpdate]);

  // ─── Mouse / touch interaction ───────────────────────────

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY };
    dragViewStart.current = {
      x: viewRef.current.centerX,
      y: viewRef.current.centerY,
    };
    const canvas = canvasRef.current;
    if (canvas) canvas.style.cursor = 'grabbing';
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    const a = universeRef.current.scaleFactor;
    const scale = viewRef.current.scale;
    viewRef.current.centerX = dragViewStart.current.x - dx / (scale * a);
    viewRef.current.centerY = dragViewStart.current.y - dy / (scale * a);
  }, []);

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      const wasDrag =
        Math.abs(e.clientX - dragStart.current.x) > 3 ||
        Math.abs(e.clientY - dragStart.current.y) > 3;
      isDragging.current = false;
      const canvas = canvasRef.current;
      if (canvas) canvas.style.cursor = 'grab';

      if (wasDrag) return;

      // Click: find nearest galaxy and recenter on it
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      const a = universeRef.current.scaleFactor;
      const view = viewRef.current;
      const w = rect.width;
      const h = rect.height;

      let bestDist = 30; // click radius in pixels
      let bestId: number | null = null;

      for (const g of galaxiesRef.current) {
        const physRelX = g.physicalX - view.centerX * a;
        const physRelY = g.physicalY - view.centerY * a;
        const sx = w / 2 + physRelX * view.scale;
        const sy = h / 2 + physRelY * view.scale;
        const dist = Math.hypot(clickX - sx, clickY - sy);
        if (dist < bestDist) {
          bestDist = dist;
          bestId = g.id;
        }
      }

      if (bestId !== null) {
        const g = galaxiesRef.current[bestId];
        // Recenter using a comoving position consistent with the galaxy's
        // current rendered physical position. For clustered/bound galaxies,
        // physical position may not equal comoving * a.
        const a = universeRef.current.scaleFactor;
        if (a > 0) {
          viewRef.current.centerX = g.physicalX / a;
          viewRef.current.centerY = g.physicalY / a;
        } else {
          viewRef.current.centerX = g.comovingX;
          viewRef.current.centerY = g.comovingY;
        }
        selectedGalaxyRef.current = bestId;
      }
    },
    []
  );

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    viewRef.current.scale = Math.max(
      2,
      Math.min(500, viewRef.current.scale * zoomFactor)
    );
  }, []);

  return (
    <canvas
      ref={canvasRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onWheel={handleWheel}
      style={{
        width: '100%',
        height: '100%',
        display: 'block',
        cursor: 'grab',
      }}
    />
  );
}
