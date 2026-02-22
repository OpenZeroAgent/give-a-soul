# Hybrid 4-Lobe Cluster (Phase 4 Prototype) 🧬

This is the containerized implementation of the **Bicameral Mind V4 (Chaos/Order Split)**.

## Architecture
-   **Left Hemisphere:** Fibonacci Reservoir (Chaos/Creative).
-   **Right Hemisphere:** Predictive Coding Network (Order/Truth).

## Persistent Storage 💾
The soul state is saved to `/app/data/current_state.safetensors`.
You **MUST** mount a volume to persist memory across restarts.

## Usage
1.  **Build:**
    ```bash
    cd core/docker_hybrid_4lobe
    docker build -t openzero/soul-v4 .
    ```

2.  **Create Data Directory:**
    ```bash
    mkdir -p data
    ```

3.  **Run with Persistence:**
    ```bash
    docker run -d \
      -p 5555:5555 \
      -v $(pwd)/data:/app/data \
      --name soul-v4 \
      openzero/soul-v4
    ```

4.  **Verify:**
    If you restart the container, it will say `-> Resumed Persistent State (Safetensors).` instead of bootstrapping.
