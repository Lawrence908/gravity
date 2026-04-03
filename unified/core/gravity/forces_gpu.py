"""GPU-accelerated gravitational forces via CuPy (optional).

Same interface as forces_cpu.compute_accelerations_vectorized.
State stays in NumPy; only the force computation runs on GPU, result copied back.
Requires: pip install cupy-cuda12x (or cupy-cuda11x for CUDA 11).
"""

from __future__ import annotations

import numpy as np

from .state import ParticleState

try:
    import cupy as cp
    from cupy_backends.cuda.api.runtime import CUDARuntimeError
except ImportError as e:
    cp = None
    CUDARuntimeError = None  # type: ignore[misc, assignment]
    _IMPORT_ERROR = e


def compute_halo_acceleration(
    positions: np.ndarray,
    M_halo: float,
    a_halo: float,
    G: float = 1.0,
) -> np.ndarray:
    """Acceleration from a static Hernquist dark-matter halo centred at the origin.

    O(N) so runs on CPU even when particle-particle forces use GPU.
    """
    r = np.sqrt(np.sum(positions * positions, axis=-1, keepdims=True))
    r_safe = np.maximum(r, 1e-12)
    acc = -G * M_halo / (r_safe + a_halo) ** 2 * (positions / r_safe)
    return acc.astype(positions.dtype)


def _raise_gpu_runtime_error(cause: BaseException) -> None:
    msg = (
        "GPU failed: {cause}. "
        "The host NVIDIA driver may be older than the CUDA version used by CuPy (e.g. cupy-cuda12x needs a driver that supports CUDA 12). "
        "Options: run without --gpu (CPU), install cupy-cuda11x for an older CUDA, or upgrade the host NVIDIA driver."
    ).format(cause=cause)
    raise RuntimeError(msg) from cause


def compute_accelerations_vectorized(
    state: ParticleState,
    softening: float = 0.05,
    G: float = 1.0,
) -> np.ndarray:
    """Compute accelerations on GPU using CuPy. Same physics as forces_cpu version.

    Raises RuntimeError if CuPy is not installed or no GPU is available.
    """
    if cp is None:
        raise RuntimeError(
            "CuPy is not installed. Install with: pip install cupy-cuda12x "
            "(or cupy-cuda11x for CUDA 11). Then retry with --gpu."
        ) from _IMPORT_ERROR

    pos = state.positions
    masses = state.masses
    dtype = pos.dtype

    try:
        # Copy to device (no persistent state; integrator/diagnostics stay on host)
        pos_d = cp.asarray(pos)
        masses_d = cp.asarray(masses)

        # dx[i, j] = pos[j] - pos[i]  -> shape (N, N, ndim)
        dx = pos_d[None, :, :] - pos_d[:, None, :]
        r2 = cp.sum(dx * dx, axis=-1) + softening * softening
        cp.fill_diagonal(r2, 1.0)
        inv_r3 = 1.0 / (r2 * cp.sqrt(r2))
        cp.fill_diagonal(inv_r3, 0.0)

        acc_d = G * (masses_d[None, :, None] * dx * inv_r3[:, :, None]).sum(axis=1)
        return cp.asnumpy(acc_d.astype(dtype))
    except Exception as e:
        is_cuda_runtime_error = (
            CUDARuntimeError is not None and isinstance(e, CUDARuntimeError)
        ) or (
            "cudaruntimeerror" in type(e).__name__.lower()
            or "cudaerror" in str(e).lower()
            or "insufficientdriver" in str(e).lower().replace(" ", "")
        )
        if is_cuda_runtime_error:
            _raise_gpu_runtime_error(e)
        raise
