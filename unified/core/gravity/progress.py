"""CLI progress reporting for demo runs (aligned format across 2D and 3D)."""

from __future__ import annotations


def report_progress(
    step: int,
    total_steps: int,
    label: str,
    *,
    pct_interval: float = 5.0,
    extra: str | None = None,
) -> bool:
    """Print a progress line at 0%, pct_interval%, ..., 100%. Returns True if a line was printed.

    label: short label for the run (e.g. "2D demo", "3D demo").
    extra: optional suffix (e.g. " E=-0.5  Lz=1.2") for diagnostics on the same line.
    """
    if total_steps <= 0:
        return False
    # Report at 0%, pct_interval%, ..., 100% (one line per bucket; 100% only when step == total_steps)
    interval = int(pct_interval)
    current_bucket = int(round(100 * step / total_steps / pct_interval)) * interval
    if step < total_steps and current_bucket >= 100:
        current_bucket = 100 - interval
    prev_bucket = (
        int(round(100 * (step - 1) / total_steps / pct_interval)) * interval
        if step > 0
        else -1
    )
    if step > 0 and step < total_steps and prev_bucket >= 100:
        prev_bucket = 100 - interval
    if step == 0 or step == total_steps or current_bucket != prev_bucket:
        pct_show = 100 if step == total_steps else current_bucket
        line = f"  [ {pct_show:3d}%] step {step:5d} / {total_steps:<5d}  ({label})"
        if extra:
            line += "  " + extra
        print(line, flush=True)
        return True
    return False
