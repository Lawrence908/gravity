"""Thin a replay to every Nth snapshot so it loads in the web viewer.

Large replays (e.g. 50k steps × 5k particles) can be 700MB+ and fail to load
in the browser. This script produces a smaller JSON by keeping every Nth
snapshot (and always the last one).

Usage (from repo root):

  # From .npz (recommended; avoids loading huge JSON)
  python tools/thin_replay.py path/to/replay.npz path/to/replay_thin.json --every 50

  # From existing .json (loads full file into memory)
  python tools/thin_replay.py path/to/replay.json path/to/replay_thin.json --every 50

  # Cap total snapshots instead (e.g. at most 2000)
  python tools/thin_replay.py replay.npz out.json --max-snapshots 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def thin_from_npz(npz_path: Path, out_path: Path, every: int, max_snapshots: int | None) -> None:
    from gravity.replay import load_replay

    data = load_replay(npz_path)
    n_snapshots = data["n_snapshots"]
    indices = _thin_indices(n_snapshots, every, max_snapshots)

    if data.get("variable_n"):
        positions = [data["positions"][i].tolist() for i in indices]
        masses = [data["masses"][i].tolist() for i in indices]
        out = {
            "positions": positions,
            "masses": masses,
            "steps": data["steps"][indices].tolist(),
            "dt": data["dt"],
            "n_snapshots": len(indices),
            "variable_n": True,
        }
    else:
        positions = data["positions"][indices]
        out = {
            "positions": positions.tolist(),
            "steps": data["steps"][indices].tolist(),
            "masses": data["masses"].tolist(),
            "dt": data["dt"],
            "n_particles": int(data["n_particles"]),
            "n_snapshots": len(indices),
            "variable_n": False,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Thinned to {len(indices)} snapshots (from {n_snapshots}) -> {out_path}")


def _json_parse_constant(s: str):
    """Allow NaN/Infinity from numpy in JSON (non-standard but sometimes emitted)."""
    if s == "NaN":
        return float("nan")
    if s in ("Infinity", "+Infinity"):
        return float("inf")
    if s == "-Infinity":
        return float("-inf")
    raise ValueError(f"Invalid JSON constant: {s!r}")


def thin_from_json(json_path: Path, out_path: Path, every: int, max_snapshots: int | None) -> None:
    with open(json_path) as f:
        data = json.load(f, parse_constant=_json_parse_constant)
    n_snapshots = data["n_snapshots"]
    indices = _thin_indices(n_snapshots, every, max_snapshots)

    positions = data["positions"]
    thinned_positions = [positions[i] for i in indices]

    steps = data["steps"]
    if hasattr(steps, "tolist"):
        steps = steps.tolist()
    thinned_steps = [steps[i] for i in indices]

    out = {
        "positions": thinned_positions,
        "steps": thinned_steps,
        "dt": data["dt"],
        "n_snapshots": len(indices),
        "variable_n": data.get("variable_n", False),
    }
    if out["variable_n"]:
        out["masses"] = [data["masses"][i] for i in indices]
    else:
        out["masses"] = data["masses"]
        out["n_particles"] = data.get("n_particles", len(data["masses"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Thinned to {len(indices)} snapshots (from {n_snapshots}) -> {out_path}")


def _thin_indices(n_snapshots: int, every: int, max_snapshots: int | None) -> list[int]:
    if max_snapshots is not None:
        if n_snapshots <= max_snapshots:
            return list(range(n_snapshots))
        # evenly sample max_snapshots indices including first and last
        step = (n_snapshots - 1) / (max_snapshots - 1)
        indices = [0] + [int(round(step * i)) for i in range(1, max_snapshots - 1)] + [n_snapshots - 1]
        return sorted(set(indices))
    indices = list(range(0, n_snapshots, every))
    if indices[-1] != n_snapshots - 1:
        indices.append(n_snapshots - 1)
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thin a replay file for loading in the web viewer.",
        epilog="Example: python tools/thin_replay.py replay.npz replay_thin.json --every 50",
    )
    parser.add_argument("input", type=Path, help="Input .npz or .json replay")
    parser.add_argument("output", type=Path, help="Output .json (thinned)")
    parser.add_argument("--every", type=int, default=None, metavar="N", help="Keep every Nth snapshot (and last)")
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        metavar="M",
        help="Keep at most M snapshots (evenly spaced including first and last)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    if args.every is None and args.max_snapshots is None:
        parser.error("specify --every N or --max-snapshots M")
    if args.every is not None and args.max_snapshots is not None:
        parser.error("specify only one of --every or --max-snapshots")

    every = args.every if args.every is not None else 1
    try:
        if args.input.suffix.lower() == ".npz":
            thin_from_npz(args.input, args.output, every, args.max_snapshots)
        elif args.input.suffix.lower() == ".json":
            thin_from_json(args.input, args.output, every, args.max_snapshots)
        else:
            print("Error: input must be .npz or .json", file=sys.stderr)
            return 1
    except json.JSONDecodeError as e:
        print(f"Error: JSON parse failed at position {e.pos} (~{e.pos // (1024*1024)} MB in): {e.msg}", file=sys.stderr)
        print("The file may be truncated (incomplete download) or contain invalid numbers.", file=sys.stderr)
        print("If you have the .npz for this run, thin from that: thin_replay.py replay.npz out.json --every 50", file=sys.stderr)
        return 1
    except MemoryError:
        print(
            "Error: not enough memory to load the file. For .json, thin from the .npz instead, or use --every with a larger N.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
