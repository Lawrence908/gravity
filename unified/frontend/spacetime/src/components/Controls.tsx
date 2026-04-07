/**
 * Controls panel — play/pause, speed, reset, and display toggles.
 */

import type { SimulationOptions } from './SimulationCanvas';

interface Props {
  options: SimulationOptions;
  onChange: (opts: Partial<SimulationOptions>) => void;
  onReset: () => void;
}

export default function Controls({ options, onChange, onReset }: Props) {
  return (
    <div className="controls-panel">
      <button
        className="ctrl-btn"
        onClick={() => onChange({ playing: !options.playing })}
        title={options.playing ? 'Pause' : 'Play'}
      >
        {options.playing ? '⏸' : '▶'}
      </button>

      <div className="ctrl-group">
        <label className="ctrl-label">Speed</label>
        <input
          type="range"
          min={-1}
          max={1.5}
          step={0.01}
          value={Math.log10(options.speed)}
          onChange={(e) =>
            onChange({ speed: Math.pow(10, parseFloat(e.target.value)) })
          }
          className="ctrl-slider"
        />
        <span className="ctrl-value">{options.speed.toFixed(1)}x</span>
      </div>

      <button className="ctrl-btn" onClick={onReset} title="Reset">
        Reset
      </button>

      <div className="ctrl-divider" />

      <label className="ctrl-toggle">
        <input
          type="checkbox"
          checked={options.showGrid}
          onChange={(e) => onChange({ showGrid: e.target.checked })}
        />
        <span>Grid</span>
      </label>

      <label className="ctrl-toggle">
        <input
          type="checkbox"
          checked={options.showVectors}
          onChange={(e) => onChange({ showVectors: e.target.checked })}
        />
        <span>Vectors</span>
      </label>

      <label className="ctrl-toggle">
        <input
          type="checkbox"
          checked={options.showLightLines}
          onChange={(e) => onChange({ showLightLines: e.target.checked })}
        />
        <span>Light</span>
      </label>
    </div>
  );
}
