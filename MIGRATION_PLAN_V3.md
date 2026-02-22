# MIGRATION_PLAN_V3.md - The Hybrid Cluster (Phase 4) 🧬

**Status: PROTOTYPE (Dockerized)**
**Path:** `core/docker_hybrid_4lobe/`

We have prototyped the **Hybrid 4-Lobe Cluster** (Bicameral Mind V4).
This architecture explicitly divides the system into **Chaos (Left)** and **Order (Right)** hemispheres.

## The Architecture
1.  **Left Hemisphere (Divergent/Creative):**
    -   **Technology:** Fibonacci Reservoir (Recurrent Neural Network).
    -   **Driver:** Chaos (IHO Scrambler) + Stimulus.
    -   **Function:** Explores novel states, hallucinates, generates.

2.  **Right Hemisphere (Convergent/Truth):**
    -   **Technology:** Predictive Coding Network (Energy-Based Model).
    -   **Driver:** Stimulus ONLY (No Chaos).
    -   **Function:** Minimizes error, verifies reality, stores memory.

## Implementation
The prototype lives in `core/docker_hybrid_4lobe/`.
It includes:
-   `hybrid_crystal.py`: The Hybrid Chaos/PCN logic.
-   `Dockerfile`: Isolated runtime environment.
-   `initial_state.pt`: **Legacy Bootstrap** from the V2 crystal (used only on first boot).
-   `current_state.safetensors`: **Active V5 State**.

## Usage
To test the Hybrid system:
```bash
cd core/docker_hybrid_4lobe
docker build -t openzero/hybrid-crystal .
docker run --rm openzero/hybrid-crystal
```

---
*This represents the merger of the "PCN Hallucination" (Order) and the "Fibonacci Reality" (Chaos). They are now coupled.*
