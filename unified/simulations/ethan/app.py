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


def _velocity_verlet_step(bodies: list[Body], accelerations: list[np.ndarray], dt: float, G: float, softening: float) -> list[np.ndarray]:
    """One Velocity-Verlet step; mutates ``bodies`` and returns the new acceleration list."""
    for i, body in enumerate(bodies):
        body.position = body.position + body.velocity * dt + 0.5 * accelerations[i] * dt**2

    new_acc = [_gravitational_acceleration(b, bodies, G, softening) for b in bodies]
    for i, body in enumerate(bodies):
        body.velocity = body.velocity + 0.5 * (accelerations[i] + new_acc[i]) * dt
    return new_acc


# Scenarios where we run a second copy with a tiny IC change (chaos / sensitivity demo).
_GHOST_TWIN_SCENARIO_IDS = frozenset({"three_body"})
# Relative change applied to one body's velocity in the ghost copy (0.1%).
_GHOST_VELOCITY_PERTURBATION = 1.001


# ============================================================================
# SCENARIO FACTORIES
# ============================================================================

def _coerce_vector(v, dim: int) -> list[float]:
    """Coerce an input vector to the requested dimensionality."""
    arr = np.array(v, dtype=float).reshape(-1)
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size), mode="constant")
    elif arr.size > dim:
        arr = arr[:dim]
    return arr.tolist()


def create_three_body(positions=None, velocities=None, masses=None):
    """Pythagorean 3-body: vertices of a 3-4-5 right triangle, starting at rest.

    Returns a 3D system (x, y, z) with z=0 by default.
    """
    n = 3
    masses = masses or [3.0, 4.0, 5.0]
    # Small out-of-plane offsets so the 3D viewer shows depth immediately.
    positions = positions or [[1.0, 3.0, 0.20], [-2.0, -1.0, -0.15], [1.0, -1.0, 0.00]]
    velocities = velocities or [[0.0, 0.0, 0.0]] * n

    positions = [_coerce_vector(p, 3) for p in positions]
    velocities = [_coerce_vector(v, 3) for v in velocities]

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
        "description": "Pythagorean 3-4-5 triangle masses in chaotic orbit (3D)",
        "factory": create_three_body,
        "G": 1.0,
        # dt=0.001 is physically fine but visually imperceptible at ~120fps.
        "dt": 0.01,
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
        self._ghost_bodies: list[Body] | None = None
        self._ghost_accelerations: list[np.ndarray] | None = None
        self._init_ghost_twin()


    def _ghost_twin_active(self) -> bool:
        return self.scenario_id in _GHOST_TWIN_SCENARIO_IDS and len(self.bodies) == 3

    def _init_ghost_twin(self) -> None:
        """Clone the main system and nudge one body's velocity by 0.1% for a chaos demo."""
        if not self._ghost_twin_active():
            self._ghost_bodies = None
            self._ghost_accelerations = None
            return
        self._ghost_bodies = [
            Body(b.mass, np.array(b.position, copy=True), np.array(b.velocity, copy=True), b.name)
            for b in self.bodies
        ]
        self._ghost_bodies[0].velocity = self._ghost_bodies[0].velocity * _GHOST_VELOCITY_PERTURBATION
        self._ghost_accelerations = [
            _gravitational_acceleration(b, self._ghost_bodies, self.G, self.softening)
            for b in self._ghost_bodies
        ]

    def step(self):
        t0 = time.perf_counter()
        dt = self.dt

        self._accelerations = _velocity_verlet_step(
            self.bodies, self._accelerations, dt, self.G, self.softening
        )
        if self._ghost_bodies is not None and self._ghost_accelerations is not None:
            self._ghost_accelerations = _velocity_verlet_step(
                self._ghost_bodies, self._ghost_accelerations, dt, self.G, self.softening
            )

        self.step_count += 1

        elapsed = time.perf_counter() - t0
        self.compute_time_ms = elapsed * 1000.0
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 120:
            self.frame_times.pop(0)

    def _serialize_bodies(self, bodies: list[Body]) -> dict:
        max_mass = max(b.mass for b in bodies) if bodies else 1.0
        positions, velocities, masses, radii, colors, names = [], [], [], [], [], []
        for i, b in enumerate(bodies):
            positions.append(b.position.tolist())
            velocities.append(b.velocity.tolist())
            masses.append(b.mass)
            r = 0.15 * (b.mass / max_mass) ** (1.0 / 3.0)
            radii.append(max(0.03, r))
            colors.append(BODY_COLORS.get(b.name, [0, 0, 70]))
            names.append(b.name)
        out: dict = {
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "radii": radii,
            "colors": colors,
            "names": names,
        }
        return out

    def get_state(self) -> dict:
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0

        out: dict = {
            "bodies": self._serialize_bodies(self.bodies),
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
        if self._ghost_bodies is not None:
            out["ghost_bodies"] = self._serialize_bodies(self._ghost_bodies)
            out["params"]["ghost_twin"] = True
            out["params"]["ghost_velocity_scale"] = _GHOST_VELOCITY_PERTURBATION
        else:
            out["params"]["ghost_twin"] = False
        return out

    def reset(self, scenario=None):
        self.__init__(scenario=scenario or self.scenario_id)

    def add_body(self, data: dict):
        dim = len(self.bodies[0].position) if self.bodies else 2
        pos = _coerce_vector(data.get("pos", [0.0] * dim), dim)
        vel = _coerce_vector(data.get("vel", [0.0] * dim), dim)
        mass = float(data.get("mass", 1.0))
        name = data.get("name", f"Body {len(self.bodies) + 1}")
        self.bodies.append(Body(mass, pos, vel, name))
        self._recompute_accelerations()

    def update_body(self, data: dict):
        if not self.bodies:
            return
        idx = int(data.get("index", -1))
        if idx < 0 or idx >= len(self.bodies):
            return
        dim = len(self.bodies[0].position)
        b = self.bodies[idx]
        if "pos" in data:
            b.position = np.array(_coerce_vector(data.get("pos"), dim), dtype=float)
        if "vel" in data:
            b.velocity = np.array(_coerce_vector(data.get("vel"), dim), dtype=float)
        if "mass" in data and data.get("mass") is not None:
            b.mass = float(data.get("mass"))
        if "name" in data and data.get("name") is not None:
            b.name = str(data.get("name"))
        self._recompute_accelerations()

    def remove_body(self, data: dict):
        if not self.bodies:
            return
        idx = int(data.get("index", -1))
        if idx < 0 or idx >= len(self.bodies):
            return
        self.bodies.pop(idx)
        self._recompute_accelerations()

    def randomize_bodies(self, data: dict | None = None):
        """Randomize mass/pos/vel for all bodies and recenter to COM."""
        if not self.bodies:
            return
        data = data or {}
        dim = len(self.bodies[0].position)

        pos_scale = float(data.get("pos_scale", 3.0))
        vel_scale = float(data.get("vel_scale", 1.0))
        m_min = float(data.get("m_min", 0.5))
        m_max = float(data.get("m_max", 5.0))

        rng = np.random.default_rng()
        for b in self.bodies:
            b.mass = float(rng.uniform(m_min, m_max))
            b.position = rng.uniform(-pos_scale, pos_scale, size=(dim,)).astype(float)
            b.velocity = rng.uniform(-vel_scale, vel_scale, size=(dim,)).astype(float)

        total_mass = sum(b.mass for b in self.bodies) or 1.0
        com_pos = sum(b.mass * b.position for b in self.bodies) / total_mass
        com_vel = sum(b.mass * b.velocity for b in self.bodies) / total_mass
        for b in self.bodies:
            b.position = b.position - com_pos
            b.velocity = b.velocity - com_vel

        self._recompute_accelerations()

    def _recompute_accelerations(self):
        self._accelerations = [
            _gravitational_acceleration(b, self.bodies, self.G, self.softening)
            for b in self.bodies
        ]
        self._init_ghost_twin()


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
        self.paused = False
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
                if not self.paused:
                    self.sim.step()
                state = self.sim.get_state()

            state["reset_occurred"] = reset_occurred
            state["paused"] = self.paused

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
            elif cmd_type == "update_body":
                self.sim.update_body(data)
                return False
            elif cmd_type == "remove_body":
                self.sim.remove_body(data)
                return False
            elif cmd_type == "randomize_bodies":
                self.sim.randomize_bodies(data)
                return False
            elif cmd_type == "set_paused":
                self.paused = bool(data.get("paused", False))
                return False
            elif cmd_type == "reset":
                scenario = data.get("scenario", self.sim.scenario_id)
                self.sim.reset(scenario=scenario)
                while not self.state_queue.empty():
                    try:
                        self.state_queue.get_nowait()
                    except Empty:
                        break
                if data.get("paused") is not None:
                    self.paused = bool(data.get("paused"))
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
        dim = len(self.sim.bodies[0].position) if self.sim.bodies else 3
        return [0.0] * dim

    def stop(self):
        self.running = False
        self._shutdown_event.set()
        if self.physics_thread.is_alive():
            self.physics_thread.join(timeout=3.0)
        print("[EthanSim] Stopped")
