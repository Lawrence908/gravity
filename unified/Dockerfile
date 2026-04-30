# ASTR 311 Unified — FastAPI + uvicorn, serves frontend and simulation API/WebSocket
#
# Base must include CUDA *runtime* libs (e.g. libnvrtc.so). Plain python:*-slim does not;
# CuPy can see the GPU but JIT fails with DynamicLibNotFoundError for libnvrtc otherwise.
#
# Use a CUDA 12.x runtime that is not newer than your host driver can support. Images like
# 12.4.x can trigger cudaErrorCompatNotSupportedOnDevice (forward compat not available on
# some GPUs / driver combos). Override when building, e.g.:
#   docker compose build --build-arg CUDA_IMAGE_TAG=12.0.1-runtime-ubuntu22.04
# If the host only supports CUDA 11.x, use 11.8.0-runtime-ubuntu22.04 and swap
# requirements-gpu.txt to cupy-cuda11x (see comment in that file).

ARG CUDA_IMAGE_TAG=12.1.1-runtime-ubuntu22.04

# --- Vite: Spacetime Expansion (production bundle in frontend/spacetime/dist) ---
FROM node:22-slim AS spacetime-builder
WORKDIR /build
COPY frontend/spacetime/package.json frontend/spacetime/package-lock.json ./
RUN npm ci
COPY frontend/spacetime/ ./
RUN npm run build

FROM nvidia/cuda:${CUDA_IMAGE_TAG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt backend/requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY . .
COPY --from=spacetime-builder /build/dist ./frontend/spacetime/dist

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "backend.controller:app", "--host", "0.0.0.0", "--port", "8000"]
