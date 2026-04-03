# Cosmic Origins sim server: serves web-viewer and runs gravity demos via API.
# Run with: docker compose up
# GPU path: base image provides CUDA runtime + NVRTC (libnvrtc.so) so CuPy can run; host provides driver.
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# So "python" and "pip" work
RUN ln -sf /usr/bin/python3 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install dependencies (same as local dev)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full repo so gravity module and tools resolve correctly
COPY . .

# Server runs from repo root; PYTHONPATH so "gravity" is importable from src
ENV PYTHONPATH=/app/src
ENV PORT=8000
EXPOSE 8000

# Run sim server (foreground); demo subprocess output streams to container logs
CMD ["python", "tools/sim_server.py"]
