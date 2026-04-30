/**
 * SidePanel — right-side configuration panel.
 *
 * Controls simulation parameters that require a reset to take effect
 * (galaxy count, cluster count, 2D/3D mode) plus a Reset button.
 */

import type { SimConfig } from '../App';

interface Props {
  simConfig: SimConfig;
  onChange: (cfg: Partial<SimConfig>) => void;
  onReset: () => void;
}

export default function SidePanel({ simConfig, onChange, onReset }: Props) {
  function toggle3D(is3D: boolean) {
    onChange({ is3D });
    onReset();
  }

  return (
    <div className="side-panel">
      <h3 className="side-panel-title">Configuration</h3>

      {/* 2D / 3D mode toggle */}
      <div className="side-row">
        <label className="side-label" style={{ marginBottom: 4 }}>Mode</label>
        <div className="mode-toggle">
          <button
            className={`mode-btn${!simConfig.is3D ? ' active' : ''}`}
            onClick={() => !simConfig.is3D || toggle3D(false)}
          >
            2D
          </button>
          <button
            className={`mode-btn${simConfig.is3D ? ' active' : ''}`}
            onClick={() => simConfig.is3D || toggle3D(true)}
          >
            3D
          </button>
        </div>
      </div>

      <div className="side-divider" />

      <div className="side-row">
        <label className="side-label">
          Galaxies
          <span className="side-value">{simConfig.galaxyCount}</span>
        </label>
        <input
          type="range"
          min={10}
          max={5000}
          step={10}
          value={simConfig.galaxyCount}
          onChange={(e) => onChange({ galaxyCount: parseInt(e.target.value) })}
          className="side-slider"
        />
      </div>

      <div className="side-row">
        <label className="side-label">
          Clusters
          <span className="side-value">{simConfig.clusterCount}</span>
        </label>
        <input
          type="range"
          min={0}
          max={12}
          step={1}
          value={simConfig.clusterCount}
          onChange={(e) => onChange({ clusterCount: parseInt(e.target.value) })}
          className="side-slider"
        />
      </div>

      <div className="side-divider" />

      <div className="side-row">
        <label className="side-label">
          Expansion Force (Ω_Λ)
          <span className="side-value">{simConfig.omegaLambda.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min={0}
          max={2}
          step={0.01}
          value={simConfig.omegaLambda}
          onChange={(e) => onChange({ omegaLambda: parseFloat(e.target.value) })}
          className="side-slider"
        />
        <p className="side-hint" style={{ marginTop: 4, marginBottom: 0 }}>
          0 = gravity wins · 0.69 = ΛCDM · &gt;1 = rapid expansion
        </p>
      </div>

      <p className="side-hint">All changes apply on reset</p>

      <button className="side-reset-btn" onClick={onReset}>
        Reset Simulation
      </button>
    </div>
  );
}
