"""Export a replay .npz file to JSON for the web viewer. Run from repo root:

    python tools/export_replay_to_json.py run.npz web-viewer/replay.json
    python tools/export_replay_to_json.py run.npz out.json --every 50   # thin to every 50th snapshot

Or: cd src && python -c "
from gravity.replay import load_replay
import json, sys
d = load_replay(sys.argv[1])
# Convert ndarrays to lists for JSON
out = {
  'positions': d['positions'].tolist(),
  'steps': d['steps'].tolist(),
  'masses': d['masses'].tolist(),
  'dt': d['dt'],
  'n_particles': d['n_particles'],
  'n_snapshots': d['n_snapshots'],
}
with open(sys.argv[2], 'w') as f:
    json.dump(out, f)
" path.npz out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from repo root or from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gravity.replay import load_replay


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export replay .npz to JSON for the web viewer.")
    parser.add_argument("input", type=Path, help="Input .npz replay")
    parser.add_argument("output", type=Path, help="Output .json")
    parser.add_argument("--every", type=int, default=None, metavar="N", help="Export every Nth snapshot only (reduces size)")
    args = parser.parse_args()

    npz_path = args.input
    json_path = args.output
    if not npz_path.exists():
        print(f"Error: not found: {npz_path}", file=sys.stderr)
        sys.exit(1)

    data = load_replay(npz_path)
    n_snapshots = data["n_snapshots"]

    if args.every is not None:
        indices = list(range(0, n_snapshots, args.every))
        if indices[-1] != n_snapshots - 1:
            indices.append(n_snapshots - 1)
        n_snapshots = len(indices)
    else:
        indices = list(range(n_snapshots))

    if data.get("variable_n"):
        positions = [data["positions"][i].tolist() for i in indices]
        masses = [data["masses"][i].tolist() for i in indices]
        steps_out = data["steps"][indices].tolist()
        out = {
            "positions": positions,
            "masses": masses,
            "steps": steps_out,
            "dt": data["dt"],
            "n_snapshots": n_snapshots,
            "variable_n": True,
        }
    else:
        out = {
            "positions": data["positions"][indices].tolist(),
            "steps": data["steps"][indices].tolist(),
            "masses": data["masses"].tolist(),
            "dt": data["dt"],
            "n_particles": int(data["n_particles"]),
            "n_snapshots": n_snapshots,
            "variable_n": False,
        }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f)
    print(f"Exported {n_snapshots} snapshots to {json_path}")


if __name__ == "__main__":
    main()
