/**
 * Top-level App component.
 *
 * Manages simulation options state and wires together the
 * SimulationCanvas and Controls components.
 */

import { useState, useCallback } from 'react';
import SimulationCanvas from './components/SimulationCanvas';
import type { SimulationOptions } from './components/SimulationCanvas';
import Controls from './components/Controls';
import SidePanel from './components/SidePanel';
import { DEFAULT_CONFIG } from './simulation/types';

export interface SimConfig {
  galaxyCount: number;
  clusterCount: number;
  is3D: boolean;
  omegaLambda: number;
}

export default function App() {
  const [options, setOptions] = useState<SimulationOptions>({
    playing: true,
    speed: 1,
    showGrid: true,
    showVectors: false,
    showLightLines: true,
  });

  const [simConfig, setSimConfig] = useState<SimConfig>({
    galaxyCount: DEFAULT_CONFIG.galaxyCount,
    clusterCount: DEFAULT_CONFIG.clusterCount,
    is3D: DEFAULT_CONFIG.is3D,
    omegaLambda: DEFAULT_CONFIG.omegaLambda,
  });

  const [resetSignal, setResetSignal] = useState(0);

  const handleOptionsChange = useCallback(
    (partial: Partial<SimulationOptions>) => {
      setOptions((prev) => ({ ...prev, ...partial }));
    },
    []
  );

  const handleSimConfigChange = useCallback(
    (partial: Partial<SimConfig>) => {
      setSimConfig((prev) => ({ ...prev, ...partial }));
    },
    []
  );

  const handleReset = useCallback(() => {
    setResetSignal((s) => s + 1);
  }, []);

  const handleStateUpdate = useCallback(() => {
    // HUD is drawn directly on canvas; this callback exists for future
    // React-rendered overlays that may need universe state.
  }, []);

  return (
    <div className="app">
      <SimulationCanvas
        options={options}
        simConfig={simConfig}
        onStateUpdate={handleStateUpdate}
        resetSignal={resetSignal}
      />
      <Controls
        options={options}
        onChange={handleOptionsChange}
        onReset={handleReset}
      />
      <SidePanel
        simConfig={simConfig}
        onChange={handleSimConfigChange}
        onReset={handleReset}
      />
    </div>
  );
}
