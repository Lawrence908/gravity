"""
backend/controller.py — Unified Simulation Controller

Combines:
  - Jasper's WebSocket real-time simulation streaming (msgpack or JSON)
  - Chris's replay REST API (list, load, delete replays; batch run queue)

Routes:
  /                      → serves frontend/index.html (static)
  /ws                    → WebSocket for real-time simulations
  /api/simulations       → list registered simulations
  /api/replays           → list / manage pre-computed replay files
  /api/replays/{name}    → DELETE a replay
  /health                → server health

Run with:
    cd unified && uvicorn backend.controller:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import json
import time
import traceback
import threading
import importlib
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# PATHS
# ============================================================================

BACKEND_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = BACKEND_DIR.parent
FRONTEND_DIR = UNIFIED_DIR / "frontend"
REPLAYS_DIR = FRONTEND_DIR / "replays"

# ============================================================================
# SIMULATION REGISTRY
# ============================================================================

SIMULATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "chrissim": {
        "id": "chrissim",
        "name": "N-Body Galaxy Simulator",
        "description": "CPU leapfrog N-body with disk/cloud ICs, halo, collisions",
        "module": "simulations.chris.app",
        "icon": "🌀",
        "tags": ["physics", "CPU", "N-body", "galaxy"],
        "scenarios": ["disk3d", "disk2d", "cloud3d", "cloud2d", "explosion3d", "explosion2d"],
    },
    "ethansim": {
        "id": "ethansim",
        "name": "Small-N Gravitational Systems",
        "description": "Chaotic three-body problem and Pluto system",
        "module": "simulations.ethan.app",
        "icon": "🪐",
        "tags": ["physics", "CPU", "small-N", "3-body", "Pluto"],
        "scenarios": ["three_body", "pluto_system"],
    },
    "jaspersim": {
        "id": "jaspersim",
        "name": "Real-Time Gravitational Curvature Simulator",
        "description": "GPU Yoshida 4th-order with GR corrections, spacetime grid",
        "module": "simulations.jasper.app",
        "icon": "⚛",
        "tags": ["physics", "GPU", "N-body", "GR", "spacetime"],
        "scenarios": ["solar_system"],
    },
    "bradsim": {
        "id": "bradsim",
        "name": "Brad's Simulator",
        "description": "Binary star demo — customise for your scenarios",
        "module": "simulations.brad.app",
        "icon": "⭐",
        "tags": ["physics", "CPU", "template"],
        "scenarios": ["binary_star"],
    },
}

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI(title="Unified Gravity Controller", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

serialization_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="serializer")

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

ACTIVE_RUNNERS: Dict[str, Any] = {}
RUNNER_LOCK = threading.Lock()

# ============================================================================
# BATCH RUN QUEUE STATE
# ============================================================================

_run_queue: list = []           # pending: {job_id, name, params}
_current_run: Optional[Dict] = None  # {job_id, name, progress_pct, step, total_steps}
_completed: list = []           # last 20 finished/failed jobs
_run_log: list = []             # last 500 stdout lines of the running job
_queue_lock = threading.Lock()
_queue_condition = threading.Condition(_queue_lock)
_worker_started = False
_MAX_COMPLETED = 20
_RUN_LOG_MAX = 500

# ============================================================================
# SIMULATION RUNNER
# ============================================================================

class SimulationRunner:
    """Wraps a simulation module's AsyncSimulator."""

    def __init__(self, sim_id: str, scenario: str | None = None):
        self.sim_id = sim_id
        reg = SIMULATION_REGISTRY[sim_id]
        mod = importlib.import_module(reg["module"])
        kwargs = {}
        if scenario:
            kwargs["scenario"] = scenario
        self.sim = mod.AsyncSimulator(**kwargs)

    def get_latest_state(self):
        return self.sim.get_latest_state()

    def send_command(self, cmd: dict):
        self.sim.send_command(cmd)

    def calculate_velocity(self, position, mass):
        return self.sim.calculate_velocity(position, mass)

    def stop(self):
        self.sim.stop()


def _stop_runner(runner_id: str):
    with RUNNER_LOCK:
        runner = ACTIVE_RUNNERS.pop(runner_id, None)
    if runner is not None:
        try:
            runner.stop()
        except Exception as e:
            print(f"[Controller] Error stopping runner {runner_id}: {e}")


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/api/simulations")
async def list_simulations():
    return {"simulations": list(SIMULATION_REGISTRY.values())}


@app.get("/health")
async def health():
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    return {
        "ok": True,
        "gpu": gpu_available,
        "active_simulations": len(ACTIVE_RUNNERS),
        "available_simulations": list(SIMULATION_REGISTRY.keys()),
    }


# ============================================================================
# REPLAY API
# ============================================================================

def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name or "replay").strip("_") or "replay"
    return safe[:200]


@app.get("/api/replays")
async def list_replays():
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    replays = []
    for f in sorted(REPLAYS_DIR.glob("*.json")):
        try:
            size_kb = f.stat().st_size // 1024
        except OSError:
            size_kb = 0
        replays.append({"name": f.stem, "file": f.name, "size_kb": size_kb})
    return replays


@app.delete("/api/replays/{name}")
async def delete_replay(name: str):
    safe_name = _sanitize_name(name)
    target = REPLAYS_DIR / f"{safe_name}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Replay not found")
    try:
        target.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True}


# ============================================================================
# BATCH RUN QUEUE
# ============================================================================

class RunPayload(BaseModel):
    name: str = "run"
    dim: str = "3d"
    ic: str = "disk"
    n: int = 500
    steps: int = 1000
    dt: float = 0.01
    r_min: float = 0.5
    r_max: float = 4.0
    replay_every: int = 20
    softening: float = 0.05
    seed: int = 42
    gpu: bool = True
    M_star: float = 1.0
    m_particle: Optional[float] = None
    collisions: bool = False
    r_collide: Optional[float] = None
    M_halo: float = 0.0
    a_halo: float = 5.0
    v_expand: float = 1.5


def _run_sim_job(job_id: str, params: dict) -> None:
    """Run a batch simulation in the worker thread and save a replay JSON."""
    global _current_run

    import numpy as np
    from core.gravity.state import ParticleState
    from core.gravity.init_conditions import (
        make_disk_2d, make_disk_3d, make_cloud_2d, make_cloud_3d,
        make_explosion_2d, make_explosion_3d,
    )
    from core.gravity.forces_cpu import (
        compute_accelerations_vectorized as _cpu_accel,
        compute_halo_acceleration as _halo_accel,
    )
    from core.gravity.integrators import leapfrog_step
    from core.gravity.collisions import resolve_collisions

    name      = params["name"]
    dim       = params.get("dim", "3d").lower()
    ic        = params.get("ic", "disk").lower()
    n         = int(params["n"])
    steps     = int(params["steps"])
    dt        = float(params["dt"])
    r_min     = float(params.get("r_min", 0.5))
    r_max     = float(params.get("r_max", 4.0))
    replay_every = int(params.get("replay_every", 20))
    softening = float(params.get("softening", 0.05))
    seed      = int(params.get("seed", 42))
    use_gpu   = bool(params.get("gpu", True))
    v_expand  = float(params.get("v_expand", 1.5))
    G         = 1.0
    M_star    = float(params.get("M_star", 1.0))
    m_particle = params.get("m_particle")
    if m_particle is not None:
        m_particle = float(m_particle)
    use_collisions = bool(params.get("collisions", False))
    M_halo    = float(params.get("M_halo", 0.0))
    a_halo    = float(params.get("a_halo", 5.0))

    def _log(msg: str) -> None:
        with _queue_lock:
            _run_log.append(msg)
            if len(_run_log) > _RUN_LOG_MAX:
                del _run_log[: len(_run_log) - _RUN_LOG_MAX]
        print(f"[Queue] {msg}", flush=True)

    # IC factory selection
    _factory_map = {
        ("disk", "3d"): make_disk_3d,
        ("disk", "2d"): make_disk_2d,
        ("cloud", "3d"): make_cloud_3d,
        ("cloud", "2d"): make_cloud_2d,
        ("explosion", "3d"): make_explosion_3d,
        ("explosion", "2d"): make_explosion_2d,
    }
    factory = _factory_map.get((ic, dim), make_disk_3d)

    factory_kwargs: dict = dict(
        n_particles=n, seed=seed, M_star=M_star, G=G,
        r_max=r_max, M_halo=M_halo, a_halo=a_halo,
    )
    if m_particle is not None:
        factory_kwargs["m_particle"] = m_particle
    if ic == "disk":
        factory_kwargs["r_min"] = r_min
    elif ic == "explosion":
        factory_kwargs["r_min"] = r_min   # explosion uses r_min as initial compact radius
        factory_kwargs["v_expand"] = v_expand

    _log(f"Initialising {n} particles  ({dim} {ic}, seed={seed})")
    try:
        state: ParticleState = factory(**factory_kwargs)
    except Exception as e:
        _log(f"ERROR building initial conditions: {e}")
        raise

    # GPU / CPU force selection
    _accel_fn = _cpu_accel
    gpu_label = "CPU"
    if use_gpu:
        try:
            from core.gravity.forces_gpu import compute_accelerations_vectorized as _gpu_accel
            _gpu_accel(state, softening=softening, G=G)   # warm-up / availability check
            _accel_fn = _gpu_accel
            gpu_label = "GPU (CuPy)"
            _log("GPU force computation enabled (CuPy)")
        except Exception as e:
            _log(f"GPU unavailable ({type(e).__name__}: {e}) — falling back to CPU")

    def _accel_with_halo(st: ParticleState) -> np.ndarray:
        acc = _accel_fn(st, softening=softening, G=G)
        if M_halo > 0:
            acc += _halo_accel(st.positions, M_halo, a_halo, G=G)
        return acc

    _log(f"Running {steps:,} steps  dt={dt}  softening={softening}  {gpu_label}")
    _log(f"Snapshot every {replay_every} steps  collisions={'on' if use_collisions else 'off'}")

    positions_list: list = []
    masses_list: list = []
    step_indices: list = []
    constant_masses = state.masses.copy()
    variable_n = False
    t_start = time.perf_counter()
    last_pct = -1

    for step_i in range(steps + 1):
        if step_i % replay_every == 0 or step_i == steps:
            positions_list.append(state.positions.tolist())
            step_indices.append(step_i)
            if use_collisions:
                masses_list.append(state.masses.tolist())
                if state.masses.shape[0] != constant_masses.shape[0]:
                    variable_n = True

        if step_i == steps:
            break

        state = leapfrog_step(state, dt, _accel_with_halo)
        if use_collisions:
            state = resolve_collisions(state)

        pct = int(100 * step_i / steps)
        if pct != last_pct and pct % 5 == 0:
            last_pct = pct
            elapsed = time.perf_counter() - t_start
            _log(f"  [{pct:3d}%] step {step_i:6d} / {steps}  ({elapsed:.1f}s)")
            with _queue_lock:
                if _current_run and _current_run.get("job_id") == job_id:
                    _current_run["progress_pct"] = pct
                    _current_run["step"] = step_i

    n_snapshots = len(positions_list)
    _log(f"Done — {n_snapshots} snapshots. Writing replay JSON…")

    if variable_n or (use_collisions and masses_list):
        out = {
            "positions": positions_list,
            "masses": masses_list,
            "steps": step_indices,
            "dt": dt,
            "n_snapshots": n_snapshots,
            "variable_n": True,
            "replay_every": replay_every,
        }
    else:
        out = {
            "positions": positions_list,
            "steps": step_indices,
            "masses": constant_masses.tolist(),
            "dt": dt,
            "n_particles": int(constant_masses.shape[0]),
            "n_snapshots": n_snapshots,
            "variable_n": False,
            "replay_every": replay_every,
        }

    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPLAYS_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    total = time.perf_counter() - t_start
    _log(f"Saved {out_path.name}  ({size_mb:.1f} MB)  total time {total:.1f}s")


def _queue_worker() -> None:
    global _current_run
    while True:
        with _queue_condition:
            while not _run_queue:
                _queue_condition.wait()
            job = _run_queue.pop(0)

        job_id = job["job_id"]
        name   = job["name"]
        params = job["params"]

        with _queue_lock:
            _run_log.clear()
            _current_run = {
                "job_id": job_id,
                "name": name,
                "progress_pct": 0,
                "step": 0,
                "total_steps": params["steps"],
            }

        ok = True
        error_msg = None
        try:
            _run_sim_job(job_id, params)
        except Exception as e:
            ok = False
            error_msg = str(e)
            print(f"[Queue] Job {job_id} failed: {e}", flush=True)
            traceback.print_exc()

        with _queue_lock:
            _current_run = None
            _completed.append({"job_id": job_id, "name": name, "ok": ok, **({"error": error_msg} if error_msg else {})})
            del _completed[:-_MAX_COMPLETED]


def _ensure_queue_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    with _queue_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_queue_worker, daemon=True, name="queue-worker")
        t.start()
        _worker_started = True


@app.post("/api/run")
async def submit_run(payload: RunPayload):
    safe_name = _sanitize_name(payload.name)
    params = payload.model_dump()
    params["name"] = safe_name

    # Cap snapshots to avoid absurdly large files
    n_snaps = 1 + params["steps"] // params["replay_every"]
    if n_snaps > 2000:
        params["replay_every"] = max(params["replay_every"], (params["steps"] + 1998) // 1999)

    job_id = str(uuid.uuid4())
    with _queue_condition:
        _run_queue.append({"job_id": job_id, "name": safe_name, "params": params})
        position = len(_run_queue) + (1 if _current_run else 0)
        _queue_condition.notify()

    _ensure_queue_worker()
    return {"ok": True, "job_id": job_id, "name": safe_name, "position_in_queue": position}


@app.get("/api/run/status")
async def run_status():
    with _queue_lock:
        running = dict(_current_run) if _current_run else None
        queue = [
            {"job_id": q["job_id"], "name": q["name"], "position": i + 1}
            for i, q in enumerate(_run_queue)
        ]
        completed = list(_completed)
    return {"running": running, "queue": queue, "completed": completed}


@app.get("/api/run/logs")
async def run_logs():
    with _queue_lock:
        lines = list(_run_log)
    return {"lines": lines}


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    sim_id: Optional[str] = None,
    scenario: Optional[str] = None,
):
    if not sim_id or sim_id not in SIMULATION_REGISTRY:
        await websocket.close(code=4004, reason=f"Unknown simulation: {sim_id}")
        return

    await websocket.accept()
    runner_id = f"{sim_id}_{id(websocket)}"
    print(f"[Controller] Client connected: sim={sim_id}, scenario={scenario}")

    runner = None
    try:
        runner = SimulationRunner(sim_id, scenario=scenario)
        with RUNNER_LOCK:
            ACTIVE_RUNNERS[runner_id] = runner

        use_msgpack = False
        try:
            import msgpack
            use_msgpack = True
        except ImportError:
            pass

        def _serialize(state):
            if use_msgpack:
                return msgpack.packb(_make_serializable(state), use_bin_type=True)
            return json.dumps(_make_serializable(state)).encode("utf-8")

        initial_state = runner.sim.sim.get_state()
        initial_packed = await asyncio.get_event_loop().run_in_executor(
            serialization_executor,
            lambda: _serialize(initial_state),
        )
        if initial_packed:
            await asyncio.wait_for(websocket.send_bytes(initial_packed), timeout=5.0)

        last_ping = time.time()

        while True:
            try:
                state = runner.get_latest_state()
                if state:
                    reset_flag = state.pop("reset_occurred", False)
                    packed = await asyncio.get_event_loop().run_in_executor(
                        serialization_executor,
                        lambda s=state: _serialize(s),
                    )
                    if packed:
                        try:
                            if reset_flag and use_msgpack:
                                import msgpack as mp
                                init_msg = mp.unpackb(packed, raw=False)
                                init_msg["type"] = "init"
                                packed = mp.packb(init_msg, use_bin_type=True)
                            elif reset_flag:
                                init_msg = json.loads(packed.decode("utf-8"))
                                init_msg["type"] = "init"
                                packed = json.dumps(init_msg).encode("utf-8")
                            await asyncio.wait_for(websocket.send_bytes(packed), timeout=1.0)
                        except asyncio.TimeoutError:
                            pass
                        except Exception as e:
                            if "closed" not in str(e).lower():
                                print(f"[Controller] Send error: {e}")
                            break

                now = time.time()
                if now - last_ping > 30:
                    try:
                        ping_msg = _serialize({"type": "ping"})
                        await asyncio.wait_for(websocket.send_bytes(ping_msg), timeout=1.0)
                        last_ping = now
                    except Exception:
                        break

                try:
                    raw_data = await asyncio.wait_for(websocket.receive(), timeout=0.001)
                    msg = _decode_ws_message(raw_data, use_msgpack)
                    if msg is None:
                        continue

                    mtype = msg.get("type")
                    if mtype == "pong":
                        pass
                    elif mtype == "stop_simulation":
                        break
                    elif mtype == "compute_velocity":
                        data = msg.get("data", {})
                        position = data.get("position", [0, 0, 0])
                        mass = float(data.get("mass", 1.0))
                        velocity = runner.calculate_velocity(position, mass)
                        response = _serialize({
                            "type": "velocity_computed",
                            "data": {
                                "velocity": velocity,
                                "requestId": data.get("requestId"),
                            },
                        })
                        await asyncio.wait_for(websocket.send_bytes(response), timeout=1.0)
                    else:
                        runner.send_command(msg)

                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    break
                except WebSocketDisconnect:
                    break
                except RuntimeError as e:
                    if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                        break
                    break
                except Exception as e:
                    if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                        break

                await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                break

    except asyncio.CancelledError:
        pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Controller] WebSocket error: {e}")
        traceback.print_exc()
    finally:
        _stop_runner(runner_id)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[Controller] Connection closed: sim={sim_id}")


# ============================================================================
# HELPERS
# ============================================================================

def _make_serializable(state: dict) -> dict:
    import numpy as np

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    return convert(state)


def _decode_ws_message(raw_data: dict, use_msgpack: bool) -> dict | None:
    try:
        if "text" in raw_data:
            msg = json.loads(raw_data["text"])
        elif "bytes" in raw_data:
            if use_msgpack:
                import msgpack
                msg = msgpack.unpackb(raw_data["bytes"], raw=True, strict_map_key=False)
                msg = _decode_bytes_keys(msg)
            else:
                msg = json.loads(raw_data["bytes"].decode("utf-8"))
        else:
            return None
    except Exception as e:
        print(f"[Controller] Decode error: {e}")
        return None

    if not isinstance(msg, dict):
        return None
    return msg


def _decode_bytes_keys(msg: dict) -> dict:
    decoded = {}
    for k, v in msg.items():
        key = k.decode("utf-8") if isinstance(k, bytes) else k
        if isinstance(v, dict):
            val = _decode_bytes_keys(v)
        elif isinstance(v, bytes):
            try:
                val = v.decode("utf-8")
            except Exception:
                val = v
        else:
            val = v
        decoded[key] = val
    return decoded


# ============================================================================
# STATIC FILE SERVING
# ============================================================================

REPLAYS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/replays", StaticFiles(directory=str(REPLAYS_DIR), check_dir=False), name="replays")


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/viewer.html")
async def serve_viewer():
    return FileResponse(FRONTEND_DIR / "viewer.html")


@app.get("/{path:path}")
async def serve_frontend(path: str):
    file_path = FRONTEND_DIR / path
    if file_path.is_file():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Not found")
