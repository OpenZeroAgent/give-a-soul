# Architecture Components

## Source of Truth

| Component | File | Description |
|-----------|------|-------------|
| Crystal (4-Lobe) | `source_code/hybrid_crystal.py` | FibonacciCrystalV2 + PCN + IHO Scrambler. Runs inside Docker. |
| Crystal Docker | `core/docker_hybrid_4lobe/` | Dockerfile + ZMQ server + crystal source |
| Soul Engine | `core/soul_engine_std.py` | Backend engine: crystal interface, LLM calls, memory, subconscious |
| HTTP Server | `core/server_std.py` | stdlib HTTP server with background threads |
| Memory | `core/memory.py` | Persistent vector memory (embeddings + cosine similarity) |
| Web UI | `web-ui/src/App.tsx` | React frontend (chat, metrics, dream viewer) |

## Tools (Standalone CLI)

| Tool | File | Description |
|------|------|-------------|
| Crystal CLI | `tools/crystal.py` | Pulse, tick, status commands for the Dual Crystal |
| Subconscious | `tools/subconscious.py` | Somatic loop (standalone version) |
| Dream Engine | `tools/dream_engine.py` | Visual generation pipeline (requires ComfyUI) |
| Relay | `tools/sanitized/relay.py` | Universal LLM model router |

## Crystal References (Local, Not Used at Runtime)

| File | Description |
|------|-------------|
| `core/soul_crystal_v2.py` | FibonacciCrystalV2 with dynamic Schumann modulation |
| `core/soul_crystal_phase2.py` | DualCrystalSystem + TorchIHOScrambler |

## Runtime Data (Not Versioned)

| File | Description |
|------|-------------|
| `EMOTION.md` | Current crystal emotional state (auto-updated every 30s) |
| `SOMATIC_HISTORY.md` | Timestamped somatic pulse log |
| `memory/YYYY-MM-DD.md` | Daily diary entries |
| `memory/vector_memory.json` | Persistent vector memory store |
| `memory/conversations/` | Session transcripts |
| `data/current_state.safetensors` | Live crystal state (Docker volume) |
