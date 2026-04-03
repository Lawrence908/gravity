"""
simulations/brad/app.py — Brad's simulation template

Starter module implementing the AsyncSimulator interface.
Modify the SCENARIOS dict and BradSimulator to create custom
initial conditions and physics variations.

Currently set up as a simple 2-body (binary star) demo.
"""

import math
import time
import threading
import numpy as np
from queue import Queue, Empty

# ============================================================================
# SCENARIOS — Add your own here
# ============================================================================

SCENARIOS = {
    "binary_star": {
        "name": "Binary Star System",
        "description": "Two equal-mass stars orbiting their common center of mass",
    },
}


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class BradSimulator:
    """Simple N-body simulator — customise this for your scenarios."""

    G = 1.0

    def __init__(self, scenario="binary_star"):
        self.scenario_id = scenario
        self.step_count = 0
        self.dt = 0.005
        self.compute_time_ms = 0.0
        self.frame_times: list[float] = []

        if scenario == "binary_star":
            self._init_binary_star()
        else:
            self._init_binary_star()

    def _init_binary_star(self):
        separation = 2.0
        mass = 1.0
        v_orbit = math.sqrt(self.G * mass / (2.0 * separation))

        self.positions = np.array([
            [-separation / 2, 0.0, 0.0],
            [separation / 2, 0.0, 0.0],
        ], dtype=float)
        self.velocities = np.array([
            [0.0, 0.0, -v_orbit],
            [0.0, 0.0, v_orbit],
        ], dtype=float)
        self.masses = np.array([mass, mass], dtype=float)
        self.names = ["Star A", "Star B"]
        self.colors = [[45, 100, 95], [200, 80, 80]]
        self.radii = [0.12, 0.12]

    @property
    def n(self):
        return len(self.masses)

    def step(self):
        t0 = time.perf_counter()

        n = self.n
        acc = np.zeros_like(self.positions)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = self.positions[j] - self.positions[i]
                r2 = np.dot(dx, dx) + 1e-6
                r = math.sqrt(r2)
                acc[i] += self.G * self.masses[j] / (r2 * r) * dx

        v_half = self.velocities + 0.5 * self.dt * acc
        self.positions = self.positions + self.dt * v_half

        acc_new = np.zeros_like(self.positions)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = self.positions[j] - self.positions[i]
                r2 = np.dot(dx, dx) + 1e-6
                r = math.sqrt(r2)
                acc_new[i] += self.G * self.masses[j] / (r2 * r) * dx

        self.velocities = v_half + 0.5 * self.dt * acc_new
        self.step_count += 1

        elapsed = time.perf_counter() - t0
        self.compute_time_ms = elapsed * 1000.0
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 120:
            self.frame_times.pop(0)

    def get_state(self) -> dict:
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        return {
            "bodies": {
                "positions": self.positions.tolist(),
                "velocities": self.velocities.tolist(),
                "masses": self.masses.tolist(),
                "radii": list(self.radii),
                "colors": list(self.colors),
                "names": list(self.names),
            },
            "params": {
                "G": self.G,
                "dt": self.dt,
                "scenario": self.scenario_id,
            },
            "performance": {
                "compute_time_ms": round(self.compute_time_ms, 3),
                "avg_fps": round(1.0 / avg_time) if avg_time > 0 else 0,
                "body_count": self.n,
            },
        }

    def reset(self, scenario=None):
        self.__init__(scenario=scenario or self.scenario_id)

    def add_body(self, data: dict):
        pos = np.array(data.get("pos", [0.0, 0.0, 0.0]), dtype=float).reshape(1, -1)
        vel = np.array(data.get("vel", [0.0, 0.0, 0.0]), dtype=float).reshape(1, -1)
        mass = float(data.get("mass", 1.0))
        name = data.get("name", f"Body {self.n + 1}")
        color = data.get("color", [0, 0, 70])
        radius = float(data.get("radius", 0.08))

        self.positions = np.vstack([self.positions, pos])
        self.velocities = np.vstack([self.velocities, vel])
        self.masses = np.append(self.masses, mass)
        self.names.append(name)
        self.colors.append(color)
        self.radii.append(radius)


# ============================================================================
# ASYNC SIMULATOR — Controller Interface
# ============================================================================

class AsyncSimulator:
    """Wraps BradSimulator in a physics thread for the unified controller."""

    def __init__(self, scenario="binary_star", fps=60):
        self.sim = BradSimulator(scenario=scenario)
        self.state_queue: Queue = Queue(maxsize=2)
        self.command_queue: Queue = Queue()
        self.running = True
        self.physics_fps = fps
        self.sim_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self.physics_thread = threading.Thread(
            target=self._physics_loop, daemon=True, name="brad-physics")
        self.physics_thread.start()
        print(f"[BradSim] Physics thread started (scenario={scenario}, fps={fps})")

    def _physics_loop(self):
        target_dt = 1.0 / self.physics_fps
        while self.running and not self._shutdown_event.is_set():
            t0 = time.perf_counter()
            reset_occurred = False

            try:
                while True:
                    cmd = self.command_queue.get_nowait()
                    with self.sim_lock:
                        if self._handle_command(cmd):
                            reset_occurred = True
            except Empty:
                pass

            with self.sim_lock:
                self.sim.step()
                state = self.sim.get_state()

            state["reset_occurred"] = reset_occurred

            try:
                self.state_queue.put_nowait(state)
            except Exception:
                pass

            sleep_time = max(0.0, target_dt - (time.perf_counter() - t0))
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[BradSim] Physics thread stopped")

    def _handle_command(self, cmd: dict) -> bool:
        try:
            cmd_type = cmd.get("type")
            data = cmd.get("data", {}) or {}

            if cmd_type == "update_params":
                if "dt" in data:
                    self.sim.dt = float(data["dt"])
                return False
            elif cmd_type == "add_body":
                self.sim.add_body(data)
                return False
            elif cmd_type == "reset":
                scenario = data.get("scenario", self.sim.scenario_id)
                self.sim.reset(scenario=scenario)
                while not self.state_queue.empty():
                    try:
                        self.state_queue.get_nowait()
                    except Empty:
                        break
                return True
            else:
                print(f"[BradSim] Unknown command: {cmd_type}")
        except Exception as e:
            print(f"[BradSim] Command error: {e}")
            import traceback
            traceback.print_exc()
        return False

    def get_latest_state(self):
        try:
            return self.state_queue.get_nowait()
        except Empty:
            return None

    def send_command(self, cmd: dict):
        self.command_queue.put(cmd)

    def calculate_velocity(self, position, mass):
        return [0.0, 0.0, 0.0]

    def stop(self):
        self.running = False
        self._shutdown_event.set()
        if self.physics_thread.is_alive():
            self.physics_thread.join(timeout=3.0)
        print("[BradSim] Stopped")
