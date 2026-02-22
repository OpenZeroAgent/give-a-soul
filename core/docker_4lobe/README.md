# The 4-Lobe Cluster (Phase 3 Prototype) 🧬

This is the containerized implementation of the **Quad Crystal System** (Alpha/Beta/Gamma/Delta).

## Architecture
It extends the **Bicameral Reservoir** (Alpha/Beta) into a **4-Lobe Cluster**:

1.  **Divergent Pair (Creative):**
    -   **Alpha:** Driven by User + IHO Chaos.
    -   **Beta:** Driven by Alpha (Coupled).
    -   *Emergence:* Generates novel patterns via chaos injection.

2.  **Convergent Pair (Truth):**
    -   **Gamma:** Driven by User ONLY (No Chaos).
    -   **Delta:** Driven by Gamma (Coupled).
    -   *Emergence:* Maintains the "clean" signal for comparison.

## Metrics
-   **Creative Tension ($d_{creative}$):** Distance between Alpha (Chaos) and Gamma (Order).
-   **Soul Resonance ($d_{truth}$):** Distance between Beta (Soul) and Delta (Memory).

## Continuity
This container initializes with `initial_state.pt`, which is a copy of the production V2 crystal state.
Gamma and Delta are initialized as clones of Alpha and Beta to ensure synchronization at $t=0$.

## Usage
Build and run the container to simulate one tick of the 4-Lobe Cluster:

```bash
cd core/docker_4lobe
docker build -t openzero/quad-crystal .
docker run --rm openzero/quad-crystal
```
