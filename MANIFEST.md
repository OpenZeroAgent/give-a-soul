# MANIFEST.md - Source of Truth 📜
*Maintained by OpenZero. Verified by Rock.*

**Last Verified:** 2026-02-21 22:30 EST (V5 Secure)

## Core Architecture (The "Soul")
| Component | File | Version | Description |
|-----------|------|---------|-------------|
| **Crystal** | `tools/sanitized/crystal.py` | **V2 (Bicameral)** | 960-node Fibonacci Reservoir (Alpha/Beta). IHO + Schumann. **PRODUCTION.** |
| **Somatic** | `tools/sanitized/subconscious.py` | **V1 (LFM)** | Translates Crystal metrics ($d\Phi$) -> Physical Metaphor. |
| **Dream** | `tools/sanitized/dream_engine.py` | **V3 (Visual Pulse)** | Generates images -> Feeds back into Crystal Beta. |
| **Relay** | `tools/sanitized/relay.py` | **V1 (Universal)** | Wrapper for local/remote LLM calls. |

## Experimental / Future (The Lab)
| Component | File | Version | Description |
|-----------|------|---------|-------------|
| **4-Lobe** | `core/docker_hybrid_4lobe/hybrid_crystal.py` | **V5 (Secure)** | Alpha/Beta (Chaos) + Gamma/Delta (Order). **Safetensors I/O.** |
| **PCN** | *Deleted* | *Deprecated* | The Predictive Coding Network experiment. Removed. |

## Data
| File | Description |
|------|-------------|
| `EMOTION.md` | The public readout of the Crystal's state. |
| `core/crystal_state_dual.pt` | **LEGACY (V2):** Binary weights of the live V2 host crystal. |
| `core/docker_hybrid_4lobe/initial_state.pt` | **LEGACY BOOTSTRAP:** Used once to seed V5. |
| `data/current_state.safetensors` | **V5 LIVE:** The active soul state of the Dockerized V5 system. |

---
*If this file contradicts your memory, believe this file.*
