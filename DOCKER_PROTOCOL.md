# Docker Protocol

## Purpose

The Crystal runs inside a Docker container to isolate PyTorch dependencies and prevent state corruption from host environment changes.

## Container Specification

| Property | Value |
|----------|-------|
| Image | `openzero/soul-v5` |
| Base | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` |
| Port | 5555 (ZMQ) |
| Protocol | ZeroMQ REQ/REP with JSON payloads |
| State format | `safetensors` (no pickle) |
| Volume | `/app/data` → `./data/` |

## Build

```bash
cd core/docker_hybrid_4lobe
docker build -t openzero/soul-v5 .
```

## Run

```bash
mkdir -p data
docker run -d -p 5555:5555 -v $(pwd)/data:/app/data --name soul-v5 openzero/soul-v5
```

## API

| Action | Payload | Response |
|--------|---------|----------|
| Pulse | `{"action": "pulse", "text_vector": [480 floats]}` | `{"status": "ok", "metrics": {...}}` |
| Status | `{"action": "status"}` | `{"status": "ok", "metrics": {...}}` |

## Stress Test

```bash
docker run --rm openzero/soul-v5 python3 stress_test_v5.py
```

Runs a 1–40 Hz frequency sweep and reports tension metrics at each step.
