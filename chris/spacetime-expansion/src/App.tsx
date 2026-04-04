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

export default function App() {
  const [options, setOptions] = useState<SimulationOptions>({
    playing: true,
    speed: 1,
    showGrid: true,
    showVectors: false,
    showLightLines: true,
  });

  const [resetSignal, setResetSignal] = useState(0);

  const handleOptionsChange = useCallback(
    (partial: Partial<SimulationOptions>) => {
      setOptions((prev) => ({ ...prev, ...partial }));
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
        onStateUpdate={handleStateUpdate}
        resetSignal={resetSignal}
      />
      <Controls
        options={options}
        onChange={handleOptionsChange}
        onReset={handleReset}
      />
    </div>
  );
}
