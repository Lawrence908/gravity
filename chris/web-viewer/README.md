# WebGL Replay Viewer

Minimal browser-based viewer for gravity simulation replays. Uses Three.js for 3D rendering, orbit controls, a timeline slider, and **play with speed** (0.5x–1000x). Works for both **2D** and **3D** replays.

## Quick start (recommended)

Run the simulation server from the **repo root**; it serves the viewer and lets you run simulations from the browser:

```bash
.venv/bin/python tools/sim_server.py
```

**Docker:** If you run everything in containers, use:

```bash
docker compose up
```

Then open **http://localhost:8000**. Saved replays are stored in a volume (`web-viewer/replays/`). To watch simulation progress (e.g. `report_progress` and step logs) while a run is in progress:

```bash
docker compose logs -f sim-server
```

Use the **control panel** (top-right):

- **Replays:** Pick a saved replay from the dropdown (or select and click **Load**). Use **Refresh** to update the list, **Delete** to remove one.
- **Run simulation:** Set dimension (2D/3D), IC (disk/cloud), particles, steps, dt, and other parameters. Enter a **Name** (or use the suggested placeholder). Click **Run simulation**; when it finishes, the new replay is saved and loaded automatically.

Replays are stored in `web-viewer/replays/` as JSON files (free-form names).

## Public showcase (view-only, no running simulations)

**`showcase.html`** is a view-only version of the viewer: same 3D replay controls (timeline, play, speed, trails, view presets) but **no “New Simulation” panel, no Run, and no Delete**. Use it to share the project publicly (e.g. portfolio) without exposing simulation creation or compute.

- **Same container:** The server serves both `index.html` (full app) and `showcase.html`. Expose only the showcase URL publicly (e.g. `https://yoursite.com/showcase.html` or `https://showcase.yoursite.com` → `/showcase.html`). In your reverse proxy (Caddy, nginx, Traefik), allow only:
  - `GET /showcase.html`
  - `GET /replays/*` and `GET /api/replays`
  - Block `GET /`, `GET /index.html`, `POST /api/run`, and `DELETE /api/replays/*` for the public host so the full app and run/delete APIs are not reachable.
- **Optional – maximum isolation:** Run a second container that serves only static files (e.g. nginx) with `showcase.html` and a copy of the replay JSONs you want to show. No Python, no `/api/run`. Point the public URL only at this container.

## Manual workflow (no server)

1. Generate a replay and export to JSON.

   **2D:**
   ```bash
   cd src && python -m gravity.demo_2d --save-replay ../outputs/runs/run.npz --replay-every 20 --no-viz --steps 500 --n 300
   ```

   **3D:**
   ```bash
   cd src && python -m gravity.demo_3d --save-replay ../outputs/runs/run3d.npz --replay-every 20 --no-viz --steps 500 --n 200
   ```

   From **repo root**, export to the viewer folder:
   ```bash
   python tools/export_replay_to_json.py outputs/runs/run.npz web-viewer/replay.json
   ```

2. Serve the `web-viewer` folder (e.g. `python -m http.server 8080` from repo root, then open `http://localhost:8080/web-viewer/` or `http://localhost:8080/` if serving from `web-viewer`).

3. **Controls:** Timeline slider to scrub; **Play** / **Pause** and **Speed** (0.5x–1000x) to animate; orbit/zoom with the mouse.
