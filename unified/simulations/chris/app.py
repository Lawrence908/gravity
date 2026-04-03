"""
simulations/chris/app.py — Chris's N-body simulation (CPU leapfrog)

Wraps the shared core/gravity engine with the AsyncSimulator interface
expected by the unified controller.  Supports disk and cloud ICs in 2D/3D,
Hernquist dark-matter halo, and optional inelastic collisions.

Scenarios are selectable via the ``reset`` command.
"""

import math
import time
import threading
import numpy as np
from queue import Queue, Empty

from core.gravity.state import ParticleState
from core.gravity.forces_cpu import compute_accelerations_vectorized, compute_halo_acceleration
from core.gravity.integrators import leapfrog_step
from core.gravity.collisions import resolve_collisions
from core.gravity.init_conditions import (
    make_disk_2d, make_cloud_2d, make_disk_3d, make_cloud_3d,
)

# ============================================================================
# SCENARIOS
# ============================================================================

SCENARIOS = {
    "disk3d": {
        "name": "3D Disk Galaxy",
        "description": "Annular disk of particles around a central star (3D)",
        "factory": "make_disk_3d",
        "defaults": {
            "n_particles": 500, "r_min": 2.0, "r_max": 15.0,
            "M_star": 1.0, "m_particle": 0.001,
            "M_halo": 10.0, "a_halo": 7.5,
            "dt": 0.02, "softening": 0.1, "collisions": False,
        },
    },
    "disk2d": {
        "name": "2D Disk Galaxy",
        "description": "Flat annular disk of particles (2D)",
        "factory": "make_disk_2d",
        "defaults": {
            "n_particles": 500, "r_min": 2.0, "r_max": 15.0,
            "M_star": 1.0, "m_particle": 0.001,
            "M_halo": 10.0, "a_halo": 7.5,
            "dt": 0.02, "softening": 0.1, "collisions": False,
        },
    },
    "cloud3d": {
        "name": "3D Particle Cloud",
        "description": "Random spherical cloud collapsing under gravity (3D)",
        "factory": "make_cloud_3d",
        "defaults": {
            "n_particles": 300, "r_max": 10.0,
            "M_star": 1.0, "m_particle": 0.002,
            "angular_fraction": 0.3,
            "dt": 0.02, "softening": 0.1, "collisions": False,
        },
    },
    "cloud2d": {
        "name": "2D Particle Cloud",
        "description": "Flat random cloud collapsing under gravity (2D)",
        "factory": "make_cloud_2d",
        "defaults": {
            "n_particles": 300, "r_max": 10.0,
            "M_star": 1.0, "m_particle": 0.002,
            "angular_fraction": 0.3,
            "dt": 0.02, "softening": 0.1, "collisions": False,
        },
    },
}

IC_FACTORIES = {
    "make_disk_2d": make_disk_2d,
    "make_disk_3d": make_disk_3d,
    "make_cloud_2d": make_cloud_2d,
    "make_cloud_3d": make_cloud_3d,
}


# ============================================================================
# CORE SIMULATION WRAPPER
# ============================================================================

class LeapfrogSimulator:
    """Step-by-step N-body sim using the shared core/gravity leapfrog engine."""

    def __init__(self, scenario="disk3d", **overrides):
        self.scenario_id = scenario
        cfg = {**SCENARIOS[scenario]["defaults"], **overrides}

        self.dt = float(cfg.get("dt", 0.02))
        self.softening = float(cfg.get("softening", 0.1))
        self.G = float(cfg.get("G", 1.0))
        self.M_halo = float(cfg.get("M_halo", 0.0))
        self.a_halo = float(cfg.get("a_halo", 5.0))
        self.use_collisions = bool(cfg.get("collisions", False))
        self.step_count = 0
        self.compute_time_ms = 0.0
        self.frame_times: list[float] = []

        factory_name = SCENARIOS[scenario]["factory"]
        factory = IC_FACTORIES[factory_name]

        factory_kwargs: dict = {}
        for key in ("n_particles", "seed", "M_star", "m_particle",
                     "r_min", "r_max", "thickness", "angular_fraction",
                     "velocity_noise", "position_noise", "M_halo", "a_halo"):
            if key in cfg:
                factory_kwargs[key] = cfg[key]
        factory_kwargs["G"] = self.G

        self.state: ParticleState = factory(**factory_kwargs)
        self.dim = self.state.positions.shape[1]

    def _accel_fn(self, st: ParticleState) -> np.ndarray:
        acc = compute_accelerations_vectorized(st, softening=self.softening, G=self.G)
        if self.M_halo > 0:
            acc += compute_halo_acceleration(
                st.positions, self.M_halo, self.a_halo, G=self.G)
        return acc

    def step(self):
        t0 = time.perf_counter()
        self.state = leapfrog_step(self.state, self.dt, self._accel_fn)
        if self.use_collisions:
            self.state = resolve_collisions(self.state)
        self.step_count += 1
        elapsed = time.perf_counter() - t0
        self.compute_time_ms = elapsed * 1000.0
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 120:
            self.frame_times.pop(0)

    def get_state(self) -> dict:
        n = self.state.positions.shape[0]
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0

        masses = self.state.masses
        star_mass = float(masses[0]) if n > 0 else 1.0

        colors = []
        names = []
        radii = []
        for i in range(n):
            if i == 0:
                colors.append([45, 100, 95])
                names.append("Star")
                radii.append(0.15)
            else:
                colors.append([210, 70, 75])
                names.append(f"p{i}")
                r = 0.15 * (float(masses[i]) / star_mass) ** (1.0 / 3.0) if star_mass > 0 else 0.04
                radii.append(max(0.02, r))

        return {
            "bodies": {
                "positions": self.state.positions.tolist(),
                "velocities": self.state.velocities.tolist(),
                "masses": masses.tolist(),
                "radii": radii,
                "colors": colors,
                "names": names,
            },
            "params": {
                "G": self.G,
                "dt": self.dt,
                "softening": self.softening,
                "M_halo": self.M_halo,
                "scenario": self.scenario_id,
            },
            "performance": {
                "compute_time_ms": round(self.compute_time_ms, 3),
                "avg_fps": round(1.0 / avg_time) if avg_time > 0 else 0,
                "body_count": n,
            },
        }

    def reset(self, scenario=None, **overrides):
        self.__init__(scenario=scenario or self.scenario_id, **overrides)

    def calculate_orbital_velocity(self, position, mass):
        pos = np.array(position, dtype=float)
        r = np.linalg.norm(pos)
        if r < 1e-10:
            return [0.0] * self.dim
        M_enc = float(self.state.masses[0])
        if self.M_halo > 0:
            M_enc += self.M_halo * r ** 2 / (r + self.a_halo) ** 2
        v = math.sqrt(self.G * M_enc / r)
        if self.dim == 2:
            return [-v * pos[1] / r, v * pos[0] / r]
        else:
            direction = np.cross([0, 0, 1], pos / r)
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.cross([0, 1, 0], pos / r)
                norm = np.linalg.norm(direction)
            direction = direction / max(norm, 1e-10)
            return (v * direction).tolist()

    def add_body(self, data: dict):
        pos = np.array(data.get("pos", [0.0] * self.dim), dtype=float).reshape(1, -1)
        vel = np.array(data.get("vel", [0.0] * self.dim), dtype=float).reshape(1, -1)
        mass = np.array([float(data.get("mass", 0.001))], dtype=float)
        self.state = ParticleState(
            positions=np.vstack([self.state.positions, pos]),
            velocities=np.vstack([self.state.velocities, vel]),
            masses=np.concatenate([self.state.masses, mass]),
        )


# ============================================================================
# ASYNC SIMULATOR — Controller Interface
# ============================================================================

class AsyncSimulator:
    """
    Wraps LeapfrogSimulator in a dedicated physics thread.
    Matches the interface expected by the unified controller.
    """

    def __init__(self, scenario="disk3d", fps=60, **kwargs):
        self.sim = LeapfrogSimulator(scenario=scenario, **kwargs)
        self.state_queue: Queue = Queue(maxsize=2)
        self.command_queue: Queue = Queue()
        self.running = True
        self.physics_fps = fps
        self.sim_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self.physics_thread = threading.Thread(
            target=self._physics_loop, daemon=True, name="chris-physics")
        self.physics_thread.start()
        print(f"[ChrisSim] Physics thread started (scenario={scenario}, fps={fps})")

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

        print("[ChrisSim] Physics thread stopped")

    def _handle_command(self, cmd: dict) -> bool:
        """Returns True on reset."""
        try:
            cmd_type = cmd.get("type")
            data = cmd.get("data", {}) or {}

            if cmd_type == "update_params":
                if "dt" in data:
                    self.sim.dt = float(data["dt"])
                if "softening" in data:
                    self.sim.softening = float(data["softening"])
                if "collisions" in data:
                    self.sim.use_collisions = bool(data["collisions"])
                return False

            elif cmd_type == "add_body":
                self.sim.add_body(data)
                return False

            elif cmd_type == "reset":
                scenario = data.get("scenario", self.sim.scenario_id)
                overrides = {k: v for k, v in data.items() if k != "scenario"}
                self.sim.reset(scenario=scenario, **overrides)
                while not self.state_queue.empty():
                    try:
                        self.state_queue.get_nowait()
                    except Empty:
                        break
                return True

            else:
                print(f"[ChrisSim] Unknown command: {cmd_type}")

        except Exception as e:
            print(f"[ChrisSim] Command error: {e}")
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
        with self.sim_lock:
            return self.sim.calculate_orbital_velocity(position, mass)

    def stop(self):
        self.running = False
        self._shutdown_event.set()
        if self.physics_thread.is_alive():
            self.physics_thread.join(timeout=3.0)
        print("[ChrisSim] Stopped")
