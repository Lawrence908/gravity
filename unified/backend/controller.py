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
        "scenarios": ["disk3d", "disk2d", "cloud3d", "cloud2d"],
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
