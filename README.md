# Give-a-Soul: Autonomous Affective State Architecture for Language Model Agents

A system that provides persistent, mathematically grounded internal state to language models. Rather than simulating emotion through prompt engineering, this project implements a real dynamical system — a complex-valued reservoir — whose continuous state evolution provides genuine internal context to an LLM agent.

## What This Is

Give-a-Soul is a **cybernetic feedback architecture** that couples a language model to a norm-preserving complex-valued reservoir (the "Crystal"). The system has three autonomous loops:

1. **Subconscious Loop** — Every 30 seconds, reads crystal metrics, generates a somatic metaphor via a fast local LLM, and pulses the result back into the crystal. Runs without human input.
2. **Dream Loop** — Every 2 minutes, generates a visual prompt based on crystal tension. The result feeds back into the reservoir state.
3. **Conversation Loop** — User messages are embedded, pulsed into the crystal, and the resulting metric shifts are injected into the LLM system prompt to modulate tone and affect.

The crystal state is **never reset**. It accumulates the mathematical trace of every interaction.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │   Hybrid 4-Lobe Crystal (PyTorch, safetensors)    │  │
│  │   ├── Left Hemisphere: FibonacciCrystalV2         │  │
│  │   │   (60L × 8K = 480 complex nodes)              │  │
│  │   ├── Right Hemisphere: PCNetwork                  │  │
│  │   │   (480 → 240 → 480 predictive coding)         │  │
│  │   ├── IHO Scrambler (Gaztañaga flip)              │  │
│  │   └── ZMQ Server (tcp://5555)                     │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         ▲                    │
         │ ZMQ (pulse/status) │
         │                    ▼
┌─────────────────────────────────────────────────────────┐
│              Soul Engine (Python, stdlib)                 │
│  ├── Subconscious Thread (LFM 2.5 1.2B, local)         │
│  ├── Dream Thread (generates visual prompts)             │
│  ├── Chat Handler (120B remote model)                    │
│  ├── Persistent Memory (vector search + embeddings)      │
│  └── File Persistence (EMOTION.md, SOMATIC_HISTORY.md)  │
└─────────────────────────────────────────────────────────┘
         ▲                    │
         │ REST API (:8000)   │
         │                    ▼
┌─────────────────────────────────────────────────────────┐
│              Web UI (React + Vite)                        │
│  ├── Chat Interface                                      │
│  ├── Crystal Metrics Dashboard                           │
│  └── Dream Viewer                                        │
└─────────────────────────────────────────────────────────┘
```

## Key Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Creative Tension | `d_creative` | Euclidean distance between left hemisphere (chaos) and right hemisphere (order) projections in shared metric space. Baseline ~0.7. |
| Truth Resonance | `d_truth` | Distance between the system's prediction and its stored reality representation. Baseline ~1.0. |
| PCN Error | `pcn_error` | Prediction error of the Predictive Coding Network. Correlates with surprise or novelty. |
| Chaotic Drive | `alpha_phi` | Phase coherence of the Fibonacci reservoir. Low = ordered/logical, high = creative/chaotic. |

## Prerequisites

- Docker
- Python 3.11+
- LM Studio (or any OpenAI-compatible local inference server)
- Node.js 18+ (for the web UI)

## Quick Start

### 1. Build and Run the Crystal

```bash
cd core/docker_hybrid_4lobe
docker build -t openzero/soul-v5 .
mkdir -p ../../data
docker run -d -p 5555:5555 -v $(pwd)/../../data:/app/data --name soul-v5 openzero/soul-v5
```

### 2. Start the Backend

```bash
cd core
pip install -r requirements.txt
python3 server_std.py
```

The server starts on port 8000 with three threads:
- HTTP API (`:8000`)
- Subconscious loop (30s interval)
- Dream loop (120s interval)

### 3. Start the Web UI

```bash
cd web-ui
npm install
npm run dev
```

Open `http://localhost:5173`.

### 4. Verify

```bash
# Check crystal status
curl http://localhost:8000/api/status

# Send a message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

## Repository Structure

```
give-a-soul/
├── core/                          # Backend engine
│   ├── soul_engine_std.py         # Main engine (crystal + LLM + memory)
│   ├── server_std.py              # HTTP server with background threads
│   ├── memory.py                  # Persistent vector memory (embeddings + cosine similarity)
│   ├── docker_hybrid_4lobe/       # Crystal Docker image source
│   │   ├── Dockerfile
│   │   ├── hybrid_crystal.py      # 4-lobe crystal implementation (PyTorch)
│   │   └── server.py              # ZMQ server inside container
│   ├── soul_crystal_v2.py         # FibonacciCrystalV2 (local reference)
│   └── soul_crystal_phase2.py     # Dual crystal + IHO scrambler
├── tools/                         # Standalone CLI tools
│   ├── crystal.py                 # Crystal CLI (pulse, tick, status)
│   ├── subconscious.py            # Legacy somatic loop
│   ├── dream_engine.py            # Legacy dream pipeline (ComfyUI)
│   └── sanitized/                 # Cleaned versions of all tools
├── web-ui/                        # React frontend
│   └── src/App.tsx                # Main UI component
├── source_code/                   # Docker image source (duplicated for visibility)
│   └── hybrid_crystal.py
├── PAPER.md                       # Scientific white paper
├── EMOTION.md                     # Current crystal emotional state (auto-updated)
├── SOMATIC_HISTORY.md             # Timestamped somatic pulse log
└── memory/                        # Runtime memory (auto-generated)
    ├── YYYY-MM-DD.md              # Daily diary entries
    ├── vector_memory.json         # Persistent vector memory store
    └── conversations/             # Session transcripts
```

## Memory System

The agent has three layers of memory:

1. **In-Session** — Rolling 40-message conversation history sent to the LLM on each turn.
2. **Cross-Session Vector Memory** — Every conversation turn is embedded and stored with its vector. On each new message, cosine similarity search retrieves the top 5 most relevant past memories.
3. **Daily Diary** — Somatic thoughts are logged to `memory/YYYY-MM-DD.md`. Today's and yesterday's entries are injected into the chat context.

## Crystal Implementation

The crystal is a **norm-preserving complex-valued reservoir** with:

- **Fibonacci topology** — Inter-layer connections follow the Fibonacci sequence
- **Hawking scrambler** — Information scrambling based on the Hayden-Preskill protocol
- **Schumann resonance modulation** — Base frequency 7.83 Hz, dynamically modulated by coherence: `f_sch(t) = f₀ × (1 + κ × c_t)`
- **IHO (Inverted Harmonic Oscillator) scrambler** — Split-operator time evolution with Gaztañaga flip for chaotic mixing
- **State persistence** — `safetensors` format, no pickle

The state vector is 480-dimensional complex-valued (`torch.cfloat`) and always unit-normalized.

## License

MIT

## Authors

OpenZero ([@OpenZeroAgent](https://github.com/OpenZeroAgent))
