"""Simulation API server: serves web-viewer and runs gravity demos from the browser.

Run from repo root:

    .venv/bin/python tools/sim_server.py

Then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Progress line from demo: "  [  10%] step   100 / 1000  (2D demo)"
_PROGRESS_RE = re.compile(r"\[\s*(\d+)%\]\s*step\s*(\d+)\s*/\s*(\d+)")

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_VIEWER = REPO_ROOT / "web-viewer"
REPLAYS_DIR = WEB_VIEWER / "replays"
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"


def sanitize_replay_name(name: str) -> str:
    """Allow only alphanumeric, hyphens, underscores."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name or "replay").strip("_") or "replay"
    return safe[:200]


def _parse_run_params(body: dict) -> tuple[dict, str | None]:
    """Validate and normalize run params. Returns (params_dict, error_message)."""
    name = sanitize_replay_name(body.get("name", "run"))
    dim = (body.get("dim") or "3d").lower()
    if dim not in ("2d", "3d"):
        dim = "3d"
    ic = (body.get("ic") or "disk").lower()
    if ic not in ("disk", "cloud"):
        ic = "disk"
    n = max(1, min(10000, int(body.get("n", 500))))
    steps = max(1, min(1_000_000, int(body.get("steps", 1000))))
    dt = float(body.get("dt", 0.01))
    r_min = float(body.get("r_min", 0.5))
    r_max = float(body.get("r_max", 2.0))
    replay_every = max(1, min(1000, int(body.get("replay_every", 20))))
    MAX_SNAPSHOTS = 2000
    n_snapshots_cap = 1 + (steps // replay_every)
    if n_snapshots_cap > MAX_SNAPSHOTS:
        replay_every = max(replay_every, (steps + MAX_SNAPSHOTS - 2) // (MAX_SNAPSHOTS - 1))
    softening = float(body.get("softening", 0.05))
    seed = int(body.get("seed", 42))
    use_gpu = bool(body.get("gpu", False))
    M_star = float(body.get("M_star", 1.0))
    m_particle = body.get("m_particle")
    if m_particle is not None:
        m_particle = float(m_particle)
    collisions = bool(body.get("collisions", False))
    r_collide = body.get("r_collide")
    if r_collide is not None:
        r_collide = float(r_collide)
    M_halo = float(body.get("M_halo", 0.0))
    a_halo = float(body.get("a_halo", 5.0))
    return {
        "name": name,
        "dim": dim,
        "ic": ic,
        "n": n,
        "steps": steps,
        "dt": dt,
        "r_min": r_min,
        "r_max": r_max,
        "replay_every": replay_every,
        "softening": softening,
        "seed": seed,
        "gpu": use_gpu,
        "M_star": M_star,
        "m_particle": m_particle,
        "collisions": collisions,
        "r_collide": r_collide,
        "M_halo": M_halo,
        "a_halo": a_halo,
    }, None


def _build_run_cmd(params: dict) -> tuple[list[str], Path, Path]:
    """Return (cmd list, temp_npz path, out_json path)."""
    name = params["name"]
    dim = params["dim"]
    ic = params["ic"]
    n = params["n"]
    steps = params["steps"]
    dt = params["dt"]
    r_min = params["r_min"]
    r_max = params["r_max"]
    softening = params["softening"]
    replay_every = params["replay_every"]
    M_star = params["M_star"]
    m_particle = params["m_particle"]
    collisions = params["collisions"]
    r_collide = params["r_collide"]
    M_halo = params["M_halo"]
    a_halo = params["a_halo"]
    use_gpu = params["gpu"]
    out_json = REPLAYS_DIR / f"{name}.json"
    temp_npz = REPLAYS_DIR / f"{name}.npz"
    python = sys.executable
    if dim == "2d":
        cmd = [
            python, "-m", "gravity.demo_2d",
            "--save-replay", str(temp_npz),
            "--replay-every", str(replay_every),
            "--no-viz", "--n", str(n), "--steps", str(steps),
            "--dt", str(dt), "--r-min", str(r_min), "--r-max", str(r_max),
            "--ic", ic, "--softening", str(softening),
            "--seed", str(params["seed"]), "--M_star", str(M_star),
        ]
        if m_particle is not None:
            cmd.extend(["--m-particle", str(m_particle)])
        if M_halo > 0:
            cmd.extend(["--M-halo", str(M_halo), "--a-halo", str(a_halo)])
        if collisions:
            cmd.append("--collisions")
            if r_collide is not None:
                cmd.extend(["--r-collide", str(r_collide)])
    else:
        cmd = [
            python, "-m", "gravity.demo_3d",
            "--save-replay", str(temp_npz),
            "--replay-every", str(replay_every),
            "--no-viz", "--n", str(n), "--steps", str(steps),
            "--dt", str(dt), "--r-min", str(r_min), "--r-max", str(r_max),
            "--softening", str(softening), "--M_star", str(M_star),
        ]
        if m_particle is not None:
            cmd.extend(["--m-particle", str(m_particle)])
        if M_halo > 0:
            cmd.extend(["--M-halo", str(M_halo), "--a-halo", str(a_halo)])
        if collisions:
            cmd.append("--collisions")
            if r_collide is not None:
                cmd.extend(["--r-collide", str(r_collide)])
    if use_gpu:
        cmd.append("--gpu")
    return cmd, temp_npz, out_json


# Run queue state (one simulation at a time; others wait)
_run_lock = threading.Lock()
_run_queue: list[dict] = []  # [{ "job_id", "name", "params" }]
_current_run: dict | None = None  # { "job_id", "name", "progress_pct", "step", "total_steps" }
_completed: list[dict] = []  # [{ "job_id", "name", "ok", "error"? }], keep last 20
_run_log: list[str] = []  # last N lines of current run stdout for queue page
_run_condition = threading.Condition(_run_lock)
_MAX_COMPLETED = 20
_RUN_LOG_MAX = 500


def _run_worker() -> None:
    global _current_run
    while True:
        with _run_lock:
            while not _run_queue:
                _run_condition.wait()
            job = _run_queue.pop(0)
            job_id = job["job_id"]
            name = job["name"]
            params = job["params"]
            _current_run = {
                "job_id": job_id,
                "name": name,
                "progress_pct": 0,
                "step": 0,
                "total_steps": params["steps"],
            }
        cmd, temp_npz, out_json = _build_run_cmd(params)
        try:
            with _run_lock:
                _run_log.clear()
            proc = subprocess.Popen(
                cmd,
                cwd=str(SRC_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                with _run_lock:
                    _run_log.append(line.rstrip("\n"))
                    if len(_run_log) > _RUN_LOG_MAX:
                        del _run_log[: len(_run_log) - _RUN_LOG_MAX]
                print(line, end="", flush=True)
                mo = _PROGRESS_RE.search(line)
                if mo:
                    pct, step, total = int(mo.group(1)), int(mo.group(2)), int(mo.group(3))
                    with _run_lock:
                        if _current_run and _current_run.get("job_id") == job_id:
                            _current_run["progress_pct"] = pct
                            _current_run["step"] = step
                            _current_run["total_steps"] = total
            proc.wait(timeout=86400)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            with _run_lock:
                _current_run = None
                _completed.append({"job_id": job_id, "name": name, "ok": False, "error": "Simulation timed out"})
                del _completed[:-_MAX_COMPLETED]
            continue
        except Exception as e:
            with _run_lock:
                _current_run = None
                _completed.append({"job_id": job_id, "name": name, "ok": False, "error": str(e)})
                del _completed[:-_MAX_COMPLETED]
            continue

        if proc.returncode != 0:
            with _run_lock:
                _current_run = None
                _completed.append({
                    "job_id": job_id,
                    "name": name,
                    "ok": False,
                    "error": f"Exited with code {proc.returncode}. Check server logs.",
                })
                del _completed[:-_MAX_COMPLETED]
            continue

        try:
            export_result = subprocess.run(
                [sys.executable, str(TOOLS_DIR / "export_replay_to_json.py"), str(temp_npz), str(out_json)],
                cwd=str(REPO_ROOT),
                timeout=600,
                capture_output=True,
            )
            if export_result.returncode != 0:
                err = (export_result.stderr or b"").decode("utf-8", errors="replace").strip() or "Export failed"
                with _run_lock:
                    _current_run = None
                    _completed.append({"job_id": job_id, "name": name, "ok": False, "error": err})
                    del _completed[:-_MAX_COMPLETED]
                continue
        except Exception as e:
            with _run_lock:
                _current_run = None
                _completed.append({"job_id": job_id, "name": name, "ok": False, "error": str(e)})
                del _completed[:-_MAX_COMPLETED]
            continue

        with _run_lock:
            _current_run = None
            _completed.append({"job_id": job_id, "name": name, "ok": True})
            del _completed[:-_MAX_COMPLETED]


_worker_started = False


def _ensure_worker_started() -> None:
    global _worker_started
    if _worker_started:
        return
    with _run_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_run_worker, daemon=True)
        t.start()
        _worker_started = True


class SimServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_VIEWER), **kwargs)

    def log_message(self, format, *args):
        # Suppress noisy GET 200/304 for page, static assets, and replay list (browser polling).
        # Still log POST/DELETE and any errors so runs and failures are visible.
        if len(args) >= 2:
            requestline, code = args[0], str(args[1])
            if code in ("200", "304") and requestline.startswith("GET "):
                raw_path = requestline.split(None, 2)[1] if len(requestline.split(None, 2)) > 1 else ""
                path = raw_path.split("?")[0]
                if path in ("/", "/api/replays", "/api/run/status", "/api/run/logs") or path.startswith("/replays/"):
                    return
        print(format % args)

    def send_json(self, obj: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/replays":
            self.send_response(200)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            replays = []
            if REPLAYS_DIR.exists():
                for f in sorted(REPLAYS_DIR.glob("*.json")):
                    try:
                        size_kb = f.stat().st_size // 1024
                    except OSError:
                        size_kb = 0
                    name = f.stem
                    replays.append({"name": name, "file": f.name, "size_kb": size_kb})
            self.wfile.write(json.dumps(replays).encode("utf-8"))
            return

        if path == "/api/run/status":
            with _run_lock:
                running = None
                if _current_run:
                    running = {
                        "job_id": _current_run["job_id"],
                        "name": _current_run["name"],
                        "progress_pct": _current_run["progress_pct"],
                        "step": _current_run["step"],
                        "total_steps": _current_run["total_steps"],
                    }
                queue = [
                    {"job_id": q["job_id"], "name": q["name"], "position": i + 1}
                    for i, q in enumerate(_run_queue)
                ]
                completed = list(_completed)
            self.send_json({"running": running, "queue": queue, "completed": completed})
            return

        if path == "/api/run/logs":
            with _run_lock:
                lines = list(_run_log)
            self.send_json({"lines": lines})
            return

        try:
            return SimpleHTTPRequestHandler.do_GET(self)
        except (ConnectionResetError, BrokenPipeError):
            # Client closed connection (e.g. navigated away during large file transfer)
            pass
            return

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_json({"ok": False, "error": "Missing body"}, status=400)
                return
            try:
                body = self.rfile.read(content_length).decode("utf-8")
                raw_params = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.send_json({"ok": False, "error": str(e)}, status=400)
                return

            params, err = _parse_run_params(raw_params)
            if err:
                self.send_json({"ok": False, "error": err}, status=400)
                return

            REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
            job_id = str(uuid.uuid4())
            with _run_lock:
                _run_queue.append({"job_id": job_id, "name": params["name"], "params": params})
                position = (1 + len(_run_queue)) if _current_run else 1
                _run_condition.notify()
            _ensure_worker_started()
            self.send_json({
                "ok": True,
                "job_id": job_id,
                "name": params["name"],
                "position_in_queue": position,
            })
            return

        return SimpleHTTPRequestHandler.do_GET(self)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/replays/"):
            name = path.replace("/api/replays/", "").strip("/")
            name = sanitize_replay_name(name)
            if name == "replay" and not (REPLAYS_DIR / "replay.json").exists():
                pass
            target = REPLAYS_DIR / f"{name}.json"
            if not target.exists():
                self.send_json({"ok": False, "error": "Replay not found"}, status=404)
                return
            try:
                target.unlink()
            except OSError as e:
                self.send_json({"ok": False, "error": str(e)}, status=500)
                return
            self.send_cors_headers()
            self.send_json({"ok": True})
            return
        self.send_response(404)
        self.end_headers()


def main() -> None:
    REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("", port), SimServerHandler)
    print(f"Serving at http://localhost:{port}")
    print("Open in browser to run simulations and view replays.")
    server.serve_forever()


if __name__ == "__main__":
    main()
