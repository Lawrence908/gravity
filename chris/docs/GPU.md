# GPU acceleration (future)

## When CPU is enough

- **Small to medium N** (hundreds to ~10k particles): the current vectorized NumPy path is simple, debuggable, and runs anywhere (including Docker without GPU). For 5k particles and 1M steps, CPU is practical for overnight runs.
- **Correctness and portability**: one code path, no device sync or driver issues. Good for development and testing.

## When GPU helps

The force step is **O(N²)** and **embarrassingly parallel**: each particle’s acceleration can be computed independently from all others. That maps well to GPU:

- **Large N** (roughly 20k+ particles): GPU memory bandwidth and many cores can give a large speedup (often 5–20× or more vs single-threaded CPU, depending on hardware).
- **Many steps at large N**: even a few × speedup saves hours on long runs.
- **Batch or interactive use**: same code, optional GPU backend when available.

So: CPU is fine for current testing and long runs at moderate N; GPU becomes attractive when you push N (and/or want faster turnaround on big runs).

## How to add GPU later

Options, in order of integration effort:

1. **CuPy** – Replace NumPy with CuPy for the position/velocity/acceleration arrays and the vectorized force computation in `forces_cpu.py`. Same API; add a `forces_gpu.py` (or a backend switch) that uses CuPy and keeps data on device. Use `cupy.asnumpy()` only when writing replays or exporting. Requires CUDA and `pip install cupy-cuda12x` (or matching CUDA version). Fall back to CPU when CuPy is not available or no GPU is present.

2. **Numba CUDA** – Implement a CUDA kernel that computes accelerations (one thread per particle or per block). More control and often faster than CuPy for this pattern, but more code and CUDA-specific. Keep state on host and transfer each step, or keep state on device for the whole run to reduce transfers.

3. **JAX** – Write the force computation in JAX and `jax.jit` it; run on GPU with `JAX_PLATFORMS=cuda`. Can use `jax.pmap` or `jax.vmap` for parallelism. Good if you already use JAX elsewhere; adds a new dependency and a different array/tracing model.

4. **Existing N-body libraries** – Integrate a library that already has GPU support (e.g. REBOUND with OpenCL, or other astrophysics codes). Highest effort but battle-tested; may dictate data layout and workflow.

## Comparison: gravitysimulator.org (Harmony of the Spheres)

[gravitysimulator.org](https://gravitysimulator.org/) is powered by [Harmony of the Spheres](https://github.com/TheHappyKoala/Harmony-of-the-Spheres): a **browser-based** Newtonian n-body simulator built with **React, Redux, and THREE.js**. The physics engine (numerical integration, force calculation) is implemented in **JavaScript** and runs on the **CPU** in the user’s browser; THREE.js is used for 3D rendering only. So their simulation is CPU-bound, same as our current setup—just a different stack (client-side JS vs our server-side Python/NumPy). They don’t use WebGPU for the n-body step. A separate project, [NBody-WebGPU](https://github.com/jrprice/NBody-WebGPU), shows that **WebGPU** can be used for GPU-accelerated n-body in the browser, but that’s a different codebase. Bottom line: the “CPU is fine for moderate scale” approach aligns with what production in-browser sims like Harmony do; GPU (for us: CuPy/Numba on the server) is an optional upgrade when we push N.

## Docker + GPU

To use GPU inside the app container you need:

1. **CuPy in the image** – e.g. `pip install cupy-cuda12x` (or `cupy-cuda11x`) in the Dockerfile or requirements so the container has the CUDA runtime used by CuPy. The container also needs CUDA libraries (e.g. `libnvrtc.so` for CuPy’s JIT); use an NVIDIA CUDA base image (e.g. `nvidia/cuda:12.2.2-devel-ubuntu22.04`) instead of `python:slim` so those libs are present.
2. **NVIDIA Container Toolkit** on the host – so the container can see the GPU (`docker run --gpus all` or Compose `deploy.resources.reservations.devices` with the NVIDIA driver).
3. **Host driver ≥ CUDA runtime** – The container does not install an NVIDIA driver; it uses the **host** driver. CuPy brings a **CUDA runtime** (e.g. CUDA 12 for `cupy-cuda12x`). That runtime requires a minimum host driver version (e.g. for CUDA 12.x, typically driver ≥ 525.60.13). If the host driver is older, you get:

   `CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version`

**If you see "insufficient driver":**

- **Run without GPU** – Don't pass `--gpu`; the app uses the CPU path and works everywhere.
- **Use an older CUDA stack** – Install `cupy-cuda11x` instead of `cupy-cuda12x` in the container; CUDA 11 has lower driver requirements and may work with your current driver.
- **Upgrade the host NVIDIA driver** – Install a driver that supports the CUDA version you use (see [NVIDIA CUDA compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).

The app turns this CUDA error into a clear `RuntimeError` with the same options so the UI or logs are easier to interpret.

### WSL2

The GPU is driven by the **Windows** NVIDIA driver; WSL2 uses it automatically. If `nvidia-smi` in WSL shows a driver version and "CUDA Version: 12.x", no driver update is needed for CUDA 12. To update the driver anyway, use Windows: [NVIDIA driver download](https://www.nvidia.com/Download/index.aspx) or GeForce Experience (updates are for the Windows side; WSL2 will use the same driver after a WSL restart).

To let **Docker** (running inside WSL2) use the GPU, install the **NVIDIA Container Toolkit** inside your WSL2 distro (e.g. Ubuntu), then restart Docker. The project's `compose.yaml` reserves one GPU via `deploy.resources.reservations.devices` so the app container can use it when you run with `--gpu`.

## Recommendation

- **Now**: Keep using CPU. It’s sufficient for 5k particles and long runs, and avoids GPU setup in Docker and on machines without a GPU.
- **Later**: When you need higher N (e.g. 20k–100k+) or faster iteration, add an optional GPU path behind a config or environment check, and keep the CPU path as the default and fallback. See `src/gravity/forces_cpu.py` for the current force interface to mirror.

### Which option to choose (when you add GPU)

**Recommendation: CuPy first.**

- **CuPy** – Smallest change: same NumPy-like API, swap arrays to device and run the existing vectorized force logic. Easy to keep CPU as fallback (try `import cupy`, else use NumPy). Fits our Python server and replay workflow; no new language or build. Best first step.
- **Numba CUDA** – Use if you need maximum performance or fine-grained control after CuPy; more code and CUDA-specific.
- **JAX** – Use if you already rely on JAX for other work (e.g. differentiability); otherwise the extra dependency and tracing model aren’t worth it just for n-body.
- **Existing libraries (e.g. REBOUND)** – Use if you want a full-featured, maintained n-body package and are willing to adapt our data layout and pipeline to theirs.
