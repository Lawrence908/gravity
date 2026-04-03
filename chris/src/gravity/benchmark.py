"""Benchmark force computation at increasing N. Run with: cd src && python -m gravity.benchmark"""

from __future__ import annotations

import time

import numpy as np

from .forces_cpu import compute_accelerations_vectorized
from .init_conditions import make_disk_2d
from .state import ParticleState


def benchmark_forces(
    n_list: list[int] | None = None,
    n_warmup: int = 2,
    n_repeat: int = 5,
) -> None:
    """Time vectorized force computation for a range of particle counts.

    Prints a table: N, mean time (ms), min time (ms), time per particle (µs).
    """
    if n_list is None:
        n_list = [500, 1000, 2000, 5000, 10_000, 20_000]

    print("Benchmark: compute_accelerations_vectorized (2D disk state)")
    print("  n_warmup={}, n_repeat={}".format(n_warmup, n_repeat))
    print(f"  {'N':>8}  {'mean_ms':>10}  {'min_ms':>10}  {'µs_per_particle':>16}")
    print("  " + "-" * 50)

    for n in n_list:
        state = make_disk_2d(n, seed=1, M_star=1.0, r_min=0.5, r_max=2.0)

        # Warmup
        for _ in range(n_warmup):
            compute_accelerations_vectorized(state, softening=0.05, G=1.0)

        times_ms = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            compute_accelerations_vectorized(state, softening=0.05, G=1.0)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(times_ms))
        min_ms = float(np.min(times_ms))
        us_per_particle = mean_ms * 1000.0 / n if n > 0 else 0.0
        print(f"  {n:>8}  {mean_ms:>10.2f}  {min_ms:>10.2f}  {us_per_particle:>16.2f}")

    print()
    print("Interpretation: O(N^2) force loop; expect time to scale roughly as N^2.")
    print("On typical CPUs, 10k–20k particles per step is feasible; 50k+ may be slow.")


def main() -> None:
    benchmark_forces()


if __name__ == "__main__":
    main()
