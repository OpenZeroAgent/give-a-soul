# DOCKER_PROTOCOL.md - Scientific Isolation Mandate 🔬

**Status: MANDATORY for all experimental research.**

## The Problem
Running high-risk scientific experiments (BCI, ZUNA, Phase 7 Crystal Tests) in the core environment pollutes the runtime, creates dependency hell (PyTorch versions), and risks destabilizing the main agent session.

## The Solution: Containerized Science
All new research projects must be developed and run within isolated Docker containers.

### Protocol
1.  **Isolate:** Create a `Dockerfile` in the project root.
2.  **Define:** Specify exact dependencies (Python 3.11+, PyTorch, numpy, scipy).
3.  **Mount:** Use volume mounts for data persistence (`-v $(pwd)/data:/app/data`).
4.  **Run:** Execute experiments inside the container (`docker run --rm ...`).

### Standard Base Image
Use `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` (or similar CPU/MPS base for Mac) as the foundation.

### Example Dockerfile
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "experiment.py"]
```

## Approved Containers
-   **`openzero/crystal-lab`**: For high-risk crystal stability tests (Phase 7/8).
-   **`openzero/zuna-bridge`**: For BCI/EEG integration (requires LSL stream).
-   **`openzero/dream-v3`**: For experimental dream engine architectures.

## Enforcement
-   **Rock (QA):** Will reject PRs that lack a Dockerfile for experimental code.
-   **Dr. Light (Architect):** Must approve the container spec.

---
*Signed, Dr. Light*
