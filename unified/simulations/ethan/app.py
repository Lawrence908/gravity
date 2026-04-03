"""
simulations/ethan/app.py — Ethan's small-N simulations (3-body, Pluto system)

Velocity-Verlet integrator for chaotic 3-body problem and the full Pluto system
(Pluto + Charon + Nix + Styx + Kerberos + Hydra).

Implements the AsyncSimulator interface for the unified controller.
"""

import math
import time
import threading
import numpy as np
from queue import Queue, Empty

G_SI = 6.67430e-11

# ============================================================================
# BODY + INTEGRATOR
# ============================================================================

class Body:
    __slots__ = ("mass", "position", "velocity", "name")

    def __init__(self, mass, position, velocity, name=""):
        self.mass = float(mass)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name


def _gravitational_acceleration(body, bodies, G, softening=0.0):
    acceleration = np.zeros_like(body.position)
    for other in bodies:
        if other is not body:
            r_vec = other.position - body.position
            r2 = np.dot(r_vec, r_vec) + softening * softening
            r = math.sqrt(r2)
            if r > 1e-10:
                acceleration += G * other.mass / (r2 * r) * r_vec
    return acceleration


# ============================================================================
# SCENARIO FACTORIES
# ============================================================================

def create_three_body(positions=None, velocities=None, masses=None):
    """Pythagorean 3-body: vertices of a 3-4-5 right triangle, starting at rest."""
    n = 3
    masses = masses or [3.0, 4.0, 5.0]
    positions = positions or [[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]]
    velocities = velocities or [[0.0, 0.0]] * n

    total_mass = sum(masses)
    com_pos = sum(m * np.array(p) for m, p in zip(masses, positions)) / total_mass
    com_vel = sum(m * np.array(v) for m, v in zip(masses, velocities)) / total_mass
    positions = [(np.array(p) - com_pos).tolist() for p in positions]
    velocities = [(np.array(v) - com_vel).tolist() for v in velocities]

    return [Body(masses[i], positions[i], velocities[i], f"Body {i+1}") for i in range(n)]


def create_pluto_system():
    """Full Pluto system in the barycenter frame with circular-orbit velocities."""
    G = G_SI
    M_pluto = 1.303e22
    M_charon = 1.586e21
    M_nix = 4.5e16
    M_styx = 7.5e15
    M_kerberos = 1.65e16
    M_hydra = 4.8e16
    M_total = M_pluto + M_charon

    d_pc = 1.9571e7
    r_pluto = M_charon * d_pc / M_total
    r_charon = M_pluto * d_pc / M_total
    v_orbital = math.sqrt(G * M_total / d_pc)
    v_pluto = M_charon / M_total * v_orbital
    v_charon = M_pluto / M_total * v_orbital

    moons = [
        ("Styx",     M_styx,     4.2656e7, 45),
        ("Nix",      M_nix,      4.8694e7, 135),
        ("Kerberos", M_kerberos, 5.7783e7, 225),
        ("Hydra",    M_hydra,    6.4738e7, 315),
    ]

    bodies = [
        Body(M_pluto,  [-r_pluto, 0], [0, -v_pluto], "Pluto"),
        Body(M_charon, [r_charon, 0], [0,  v_charon], "Charon"),
    ]
    for name, mass, r, angle_deg in moons:
        a = math.radians(angle_deg)
        v = math.sqrt(G * M_total / r)
        bodies.append(Body(
            mass,
            [r * math.cos(a), r * math.sin(a)],
            [-v * math.sin(a), v * math.cos(a)],
            name,
        ))
    return bodies


BODY_COLORS = {
    "Body 1":   [0, 70, 65],
    "Body 2":   [120, 70, 55],
    "Body 3":   [240, 70, 65],
    "Pluto":    [30, 40, 75],
    "Charon":   [200, 50, 65],
    "Styx":     [60, 60, 70],
    "Nix":      [180, 50, 70],
    "Kerberos": [300, 50, 65],
    "Hydra":    [150, 60, 60],
}

SCENARIOS = {
    "three_body": {
        "name": "Chaotic Three-Body Problem",
        "description": "Pythagorean 3-4-5 triangle masses in chaotic orbit (2D)",
        "factory": create_three_body,
        "G": 1.0,
        "dt": 0.001,
        "softening": 0.01,
    },
    "pluto_system": {
        "name": "Pluto System",
        "description": "Pluto + Charon + 4 moons in SI units (2D)",
        "factory": create_pluto_system,
        "G": G_SI,
        "dt": 100.0,
        "softening": 1e5,
    },
}


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SmallNSimulator:
    """Velocity-Verlet integrator for small-N gravitational systems."""

    def __init__(self, scenario="three_body"):
        self.scenario_id = scenario
        cfg = SCENARIOS[scenario]
        self.bodies = cfg["factory"]()
        self.G = cfg["G"]
        self.dt = cfg["dt"]
        self.softening = cfg["softening"]
        self.step_count = 0
        self.compute_time_ms = 0.0
        self.frame_times: list[float] = []

        self._accelerations = [
            _gravitational_acceleration(b, self.bodies, self.G, self.softening)
            for b in self.bodies
        ]

    def step(self):
        t0 = time.perf_counter()
        dt = self.dt

        for i, body in enumerate(self.bodies):
            body.position += body.velocity * dt + 0.5 * self._accelerations[i] * dt ** 2

        new_acc = [
            _gravitational_acceleration(b, self.bodies, self.G, self.softening)
            for b in self.bodies
        ]
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * (self._accelerations[i] + new_acc[i]) * dt

        self._accelerations = new_acc
        self.step_count += 1

        elapsed = time.perf_counter() - t0
        self.compute_time_ms = elapsed * 1000.0
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 120:
            self.frame_times.pop(0)

    def get_state(self) -> dict:
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        max_mass = max(b.mass for b in self.bodies) if self.bodies else 1.0

        positions, velocities, masses, radii, colors, names = [], [], [], [], [], []
        for b in self.bodies:
            positions.append(b.position.tolist())
            velocities.append(b.velocity.tolist())
            masses.append(b.mass)
            r = 0.15 * (b.mass / max_mass) ** (1.0 / 3.0)
            radii.append(max(0.03, r))
            colors.append(BODY_COLORS.get(b.name, [0, 0, 70]))
            names.append(b.name)

        return {
            "bodies": {
                "positions": positions,
                "velocities": velocities,
                "masses": masses,
                "radii": radii,
                "colors": colors,
                "names": names,
            },
            "params": {
                "G": self.G,
                "dt": self.dt,
                "scenario": self.scenario_id,
            },
            "performance": {
                "compute_time_ms": round(self.compute_time_ms, 3),
                "avg_fps": round(1.0 / avg_time) if avg_time > 0 else 0,
                "body_count": len(self.bodies),
            },
        }

    def reset(self, scenario=None):
        self.__init__(scenario=scenario or self.scenario_id)

    def add_body(self, data: dict):
        dim = len(self.bodies[0].position) if self.bodies else 2
        pos = data.get("pos", [0.0] * dim)
        vel = data.get("vel", [0.0] * dim)
        mass = float(data.get("mass", 1.0))
        name = data.get("name", f"Body {len(self.bodies) + 1}")
        self.bodies.append(Body(mass, pos, vel, name))
        self._accelerations.append(
            _gravitational_acceleration(self.bodies[-1], self.bodies, self.G, self.softening)
        )


# ============================================================================
# ASYNC SIMULATOR — Controller Interface
# ============================================================================

class AsyncSimulator:
    """Wraps SmallNSimulator in a physics thread for the unified controller."""

    def __init__(self, scenario="three_body", fps=120):
        self.sim = SmallNSimulator(scenario=scenario)
        self.state_queue: Queue = Queue(maxsize=2)
        self.command_queue: Queue = Queue()
        self.running = True
        self.physics_fps = fps
        self.sim_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self.physics_thread = threading.Thread(
            target=self._physics_loop, daemon=True, name="ethan-physics")
        self.physics_thread.start()
        print(f"[EthanSim] Physics thread started (scenario={scenario}, fps={fps})")

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

        print("[EthanSim] Physics thread stopped")

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
                print(f"[EthanSim] Unknown command: {cmd_type}")
        except Exception as e:
            print(f"[EthanSim] Command error: {e}")
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
        return [0.0, 0.0]

    def stop(self):
        self.running = False
        self._shutdown_event.set()
        if self.physics_thread.is_alive():
            self.physics_thread.join(timeout=3.0)
        print("[EthanSim] Stopped")
