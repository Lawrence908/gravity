# ASTR 311 Unified — FastAPI + uvicorn, serves frontend and simulation API/WebSocket
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full unified app (backend, frontend, core, simulations)
COPY . .

ENV PORT=8000
EXPOSE 8000

# Serve static frontend and run API/WebSocket
CMD ["uvicorn", "backend.controller:app", "--host", "0.0.0.0", "--port", "8000"]
