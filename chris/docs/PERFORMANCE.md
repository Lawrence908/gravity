# Performance and Scaling

## Benchmarking

Run the force-computation benchmark from the project root:

```bash
cd src && python -m gravity.benchmark
```

This times `compute_accelerations_vectorized` for N = 500, 1000, 2000, 5000, 10000, 20000 (configurable in the script). The routine is O(N²) in particle count; expect runtime per step to scale roughly as N².

## CPU particle limits (reference)

The vectorized NumPy implementation uses O(N²) memory and work per step. On the project VM (see `docs/project-outline.md`):

- **Hardware:** Xeon E5-2643 v4 @ 3.40GHz, 24 logical cores, 31 GB RAM, NVIDIA RTX A2000 12GB
- **Practical range:** 10k–20k particles per step runs in a few hundred ms per step on typical single-threaded runs; 50k+ becomes slow without further optimization (e.g. Numba JIT or GPU).
- **Memory:** State and force arrays for N particles use on the order of N² * 8 bytes for the distance/force matrices; 20k particles implies ~3 GB for the largest arrays, which is within the 31 GB RAM.

Run `gravity.benchmark` on your machine to obtain actual timings and adjust target particle counts accordingly.

## Optional: GPU and Numba

GPU acceleration (e.g. Numba CUDA or CuPy) and Numba `@njit` for the loop-based force kernel are optional extensions. Optimize only after the CPU reference is correct and profiled.

**When to consider GPU:** For large N (roughly 20k+ particles), a GPU can give a substantial speedup; for N in the 1k–10k range, CPU is usually sufficient. See **docs/GPU.md** for when GPU helps and how to add it later.
