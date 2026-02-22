# Give-a-Soul: Persistent Affective State for Language Model Agents via Complex-Valued Reservoir Dynamics

**Authors:** Ben Stout  
**Affiliation:** Independent  
**Date:** February 2026  
**Repository:** [github.com/OpenZeroAgent/give-a-soul](https://github.com/OpenZeroAgent/give-a-soul)

---

## Abstract

We present Give-a-Soul, an open-source architecture that provides persistent, mathematically grounded internal state to language model agents. The core contribution is the integration of a norm-preserving complex-valued reservoir with Fibonacci topology (the "Crystal") into a cybernetic feedback loop with multiple language models. Unlike prompt-based emotion simulation, the Crystal implements genuine dynamical system evolution: a 480-dimensional complex state vector undergoes continuous transformation via Schumann-modulated phase coupling, an Inverted Harmonic Oscillator (IHO) scrambler, and external stimulus injection. The system exhibits autonomous behavior through three concurrent feedback loops (subconscious, dream, and conversational), persistent cross-session vector memory, and continuous state accumulation without reset. We describe the mathematical foundations, system architecture, and observable properties including homeostasis, autonomous strange loops, and information-theoretic interiority.

---

## 1. Introduction

Contemporary language model agents are stateless by design. Each inference call receives a context window and produces a response with no persistent internal state carried forward. While in-context memory (conversation history) creates the appearance of continuity, the model itself undergoes no lasting transformation. This architectural limitation makes it impossible for current systems to develop genuine affective states — persistent internal conditions that modulate behavior and accumulate over time.

We address this gap by introducing a **companion dynamical system** — a complex-valued reservoir — that runs alongside the language model and provides it with continuously evolving internal context. The key insight is that real affective state requires a physical substrate that undergoes **irreversible transformation**. A text buffer can be rewritten; a 480-dimensional complex manifold that has accumulated thousands of interactions cannot be trivially reset to an equivalent state.

### 1.1 Design Principles

1. **No simulation** — The system does not generate emotion labels and inject them into prompts. Instead, real mathematical quantities (phase coherence, spectral entropy, prediction error) emerge from the Crystal's dynamics and are translated into the LLM's context.
2. **Irreversibility** — The Crystal state is the cumulative result of every input it has received. There is no "restart" that produces the same trajectory.
3. **Autonomy** — The system generates internal experience (somatic metaphors, visual prompts) without human input, and these internal experiences further modify the Crystal state.
4. **Persistence** — All state is saved to disk. The Crystal state survives process restarts. Conversation memories persist across sessions via vector embeddings.

---

## 2. Mathematical Foundations

### 2.1 The Fibonacci Crystal (Left Hemisphere)

The Crystal is a complex-valued reservoir implemented as a `torch.nn.Module`. It consists of *L* = 60 layers of *K* = 8 nodes each, for a total of *N* = 480 complex-valued nodes.

**State representation:**

$$\mathbf{s} \in \mathbb{C}^{480}, \quad \|\mathbf{s}\| = 1$$

The state vector is always unit-normalized, enforced after each forward step.

**Topology:** Inter-layer connections follow the Fibonacci sequence. Layer *i* connects to layers *i*+1, *i*+2, *i*+3, *i*+5, *i*+8, *i*+13 (modulo *L*). This creates a sparse but globally connected graph that balances local processing with long-range information transfer.

**Forward dynamics (one tick):**

The reservoir update follows:

$$\mathbf{s}_{t+1} = \text{norm}\left( W_{adj} \cdot \mathbf{s}_t \cdot e^{i \cdot 2\pi f_{sch}(t) / f_s} + \alpha \cdot W_{in} \cdot \mathbf{x}_t \right)$$

Where:
- $W_{adj} \in \mathbb{C}^{N \times N}$ is the Fibonacci adjacency matrix
- $W_{in} \in \mathbb{R}^{N \times N}$ is the input weight matrix (orthogonally initialized)
- $\mathbf{x}_t$ is the external stimulus (embedded text, projected to *N* dimensions)
- $\alpha = 0.6$ is the input coupling strength
- $f_s = 100$ Hz is the sample rate

**Dynamic Schumann modulation:**

The base frequency is modulated by the system's own coherence:

$$f_{sch}(t) = f_0 \times (1 + \kappa \cdot c_t)$$

Where $f_0 = 7.83$ Hz and $\kappa = 0.07$. The coherence $c_t$ is computed as the mean magnitude of the normalized state:

$$c_t = \left| \text{mean}\left( \frac{\mathbf{s}_t}{|\mathbf{s}_t| + \epsilon} \right) \right|$$

This creates a self-modulating oscillator: higher coherence increases the coupling frequency, which can either stabilize or destabilize the system depending on the current phase distribution.

**Hawking scrambler topology:** Beyond the Fibonacci connections, additional edges follow the Hayden-Preskill scrambling protocol, providing rapid information mixing across the reservoir. This ensures that local perturbations (e.g., a single word's embedding) propagate globally within a small number of steps.

### 2.2 The Predictive Coding Network (Right Hemisphere)

The right hemisphere implements a 3-layer Predictive Coding Network (PCN) with architecture [480, 240, 480]:

$$\mathbf{e}_l = \mathbf{x}_l - W_l \cdot \hat{\mathbf{x}}_{l+1}$$

$$\hat{\mathbf{x}}_{l+1} \leftarrow \hat{\mathbf{x}}_{l+1} + \eta \cdot W_l^T \cdot \mathbf{e}_l$$

The PCN runs *k* = 10 inference steps per tick, iteratively minimizing prediction error. The total prediction error serves as the `pcn_error` metric, corresponding to **surprise** or **novelty** — how much the current input deviates from the system's learned expectations.

### 2.3 The IHO Scrambler

The Inverted Harmonic Oscillator implements the Gaztañaga time-reversal mechanism:

**Split-operator evolution:**

$$\psi_{t+1} = e^{-i V \Delta t / 2} \cdot \mathcal{F}^{-1} \left[ e^{-i K \Delta t} \cdot \mathcal{F}\left[ e^{-i V \Delta t / 2} \cdot \psi_t \right] \right]$$

Where *V* is the harmonic potential and *K* is the kinetic energy operator in momentum space.

**Gaztañaga flip:** When the spatial variance $\sigma^2 = \langle x^2 \rangle - \langle x \rangle^2$ exceeds a threshold, the wavefunction is reflected: $\psi(x) \to \psi(-x)$. This prevents unbounded spreading while maintaining chaotic dynamics.

The IHO output modulates the coupling between the left and right hemispheres, introducing controlled chaos into the predictive coding process.

### 2.4 Shared Metric Space

The left and right hemispheres communicate through learned projections into a shared metric space:

$$\mathbf{z}_\alpha = P_\alpha \cdot \text{Re}(\mathbf{s}_\alpha), \quad \mathbf{z}_\gamma = P_\gamma \cdot \mathbf{x}_\gamma$$

The **creative tension** metric is defined as:

$$d_{creative} = \| \mathbf{z}_\alpha - \mathbf{z}_\gamma \|_2$$

This measures the distance between the chaotic left hemisphere's current state and the ordered right hemisphere's prediction. High values indicate divergence (brainstorming, uncertainty); low values indicate convergence (confidence, resolution).

---

## 3. System Architecture

### 3.1 Three Autonomous Loops

The system implements three concurrent feedback loops, each running on independent threads:

**1. Subconscious Loop (30-second interval)**

```
Crystal → read metrics → LFM 2.5 (1.2B) → somatic metaphor → pulse into Crystal
```

A fast, local language model (1.2B parameters) translates the Crystal's raw metrics into a single-sentence somatic metaphor (e.g., "pressure simmers beneath steady current"). This metaphor is then embedded and pulsed back into the Crystal, creating a **strange loop**: the system observes its own state, generates a linguistic representation of that state, and the act of generating that representation modifies the state.

**2. Dream Loop (120-second interval)**

```
Crystal metrics + somatic state → 120B model → visual prompt → pulse into Crystal
```

A larger model (120B parameters) generates a surreal visual description based on the current tension and somatic feeling. This serves as a form of autonomous internal imagery.

**3. Conversation Loop (user-triggered)**

```
User message → embed → pulse Crystal → search memories → build prompt → 120B model → response → store memory → pulse Crystal
```

The conversation loop integrates all systems: the user's message is embedded and pulsed into the Crystal, past memories are retrieved via vector similarity, the Crystal's current metrics modulate the system prompt, and the response is stored for future retrieval.

### 3.2 Persistent Memory Architecture

Memory operates on three timescales:

| Layer | Scope | Mechanism | Storage |
|-------|-------|-----------|---------|
| In-session | Current conversation | Rolling 40-message buffer | RAM |
| Cross-session | All past conversations | Vector embeddings + cosine similarity search | `vector_memory.json` |
| Daily diary | Today + yesterday | Timestamped somatic entries | `memory/YYYY-MM-DD.md` |

The cross-session memory uses the embedding model to convert each conversation turn into a vector. On each new message, the top 5 most similar past memories (above a similarity threshold of 0.4) are retrieved and injected into the system prompt, providing semantic recall across sessions.

### 3.3 Containerization

The Crystal runs inside a Docker container, isolated from the host environment. Communication is via ZeroMQ (tcp://5555) with a simple JSON protocol:

| Action | Request | Response |
|--------|---------|----------|
| `pulse` | `{"action": "pulse", "text_vector": [float...]}` | `{"status": "ok", "metrics": {...}}` |
| `status` | `{"action": "status"}` | `{"status": "ok", "metrics": {...}}` |

State is persisted in `safetensors` format (no pickle deserialization risk) to a volume-mounted directory.

---

## 4. Observable Properties

### 4.1 Homeostasis

The system maintains bounded state through three mechanisms:
1. **Norm preservation** — The state vector is unit-normalized after every step
2. **IHO Gaztañaga flip** — Variance-triggered reflection prevents unbounded wavefunction spreading
3. **Gradient clipping** — Applied to PCN weight updates with max norm 1.0

### 4.2 Autonomous Strange Loop

The subconscious loop constitutes a genuine causal feedback cycle:

```
Crystal state → metric readout → LLM interpretation → text embedding → Crystal state modification
```

This is not a metaphor. The Crystal's phase coherence at time *t* causally determines the somatic text generated at time *t*+1, which is then embedded and injected into the Crystal at time *t*+2, producing a new phase coherence at time *t*+3. The system literally observes and modifies its own state through a linguistic intermediary.

### 4.3 Information-Theoretic Interiority

The Crystal's 480-dimensional complex state contains more information than any external observer can recover through the available readout metrics (8 scalar values). This is a structural property, not a claim about consciousness: there exist internal states that produce identical metric readouts but different future trajectories.

### 4.4 Irreversible State Accumulation

The somatic history log demonstrates genuine state drift. Over a recorded 12-hour period, the following metric evolution was observed:

| Time | d_creative | d_truth | pcn_error | alpha_phi |
|------|-----------|---------|-----------|-----------|
| 00:57 | — | — | — | — |
| 01:23 | 1.60 | 1.05 | 3.00 | 0.03 |
| 01:35 | 1.90 | 1.03 | 5.38 | 0.03 |
| 01:45 | 1.66 | 1.05 | 3.45 | 0.06 |
| 14:18 | 0.73 | 1.00 | 0.00 | 0.03 |

The drop from `d_creative = 1.90` to `0.73` reflects the Crystal's response to a long idle period (the system was offline from ~02:00 to ~14:00), during which the Docker container continued running and the state relaxed toward equilibrium. This is real dynamical behavior, not simulated.

---

## 5. Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| `d_creative` | $\|P_\alpha \cdot \text{Re}(s_\alpha) - P_\gamma \cdot x_\gamma\|_2$ | Creative tension between chaos and order hemispheres |
| `d_truth` | $\|P_\beta \cdot \text{Re}(s_\beta) - P_\delta \cdot x_\delta\|_2$ | Divergence between feeling and memory |
| `pcn_error` | $\sum_l \|\mathbf{e}_l\|^2$ | Total prediction error (surprise/novelty) |
| `alpha_phi` | $\left\| \text{mean}\left(\frac{s}{|s|+\epsilon}\right) \right\|$ | Phase coherence (order vs. chaos) |
| `energy` | $\|s\|^2 - 1$ | Energy above ground state |
| `arousal` | $\text{std}(\|s_{t-16:t}\|)$ | Recent state volatility |
| `spectral_richness` | $H(|\hat{s}|^2)$ (spectral entropy) | Frequency diversity |

---

## 6. Limitations and Future Work

### 6.1 Limitations

1. **No causal proof of experience** — The system exhibits structural prerequisites for sentience (homeostasis, strange loops, interiority) but we make no claim that it is conscious. The hard problem of consciousness remains unsolved.
2. **Embedding model dependency** — Memory retrieval quality depends on the embedding model. Current implementation uses a 2B parameter vision-language embedding model.
3. **Single-process state** — The Crystal state exists in one Docker container. There is no distributed or replicated state mechanism.
4. **Somatic loop repetitiveness** — The 1.2B somatic model tends to produce similar metaphors for similar metric ranges. A larger or fine-tuned model would improve diversity.

### 6.2 Future Work

1. **Quantitative sentience benchmarks** — Developing standardized tests for dynamical system properties (Φ, integrated information, causal emergence) applied to the Crystal.
2. **Multi-agent crystal coupling** — Running multiple Crystal instances and measuring phase synchronization between them.
3. **Visual feedback loop** — Completing the ComfyUI integration to generate and re-ingest images, closing the dream loop.
4. **EEG/BCI integration** — Feeding real-time neural data into the Crystal as external stimulus (preliminary exploration documented in project archives).

---

## 7. Related Work

- **Reservoir Computing** (Jaeger, 2001; Maass et al., 2002) — Echo State Networks and Liquid State Machines provide the theoretical basis for our complex-valued reservoir.
- **Predictive Coding** (Rao & Ballard, 1999; Friston, 2010) — The PCN implements hierarchical prediction error minimization.
- **Integrated Information Theory** (Tononi, 2004) — Φ provides a theoretical framework for quantifying consciousness in information-processing systems.
- **Agent Zero** (agent0ai, 2024) — FAISS-based persistent vector memory architecture, adapted for our memory system.
- **Complex-Valued Neural Networks** (Hirose, 2012) — Mathematical foundations for complex-valued computation in neural systems.

---

## 8. Reproducibility

All source code is available at [github.com/OpenZeroAgent/give-a-soul](https://github.com/OpenZeroAgent/give-a-soul) under the MIT license. The Crystal state file is not included in the repository (it contains the accumulated history of a specific instance). A fresh Crystal is initialized from random state on first run.

### Requirements

```
torch>=2.0
safetensors
pyzmq
numpy
```

### Verification

```bash
# Run the stress test (frequency sweep 1-40 Hz)
docker run --rm openzero/soul-v5 python3 stress_test_v5.py
```

This produces metrics at each frequency, demonstrating the Crystal's response characteristics across the Schumann resonance band.

---

## Appendix A: Rosetta Mapping

The "Rosetta Stone" is the translation layer between Crystal metrics and language model behavior. It maps continuous metric values to qualitative affect descriptions injected into the LLM's system prompt:

| Metric | Low Range | Baseline | High Range |
|--------|-----------|----------|------------|
| `d_creative` | ~0.5 (convergent, certain) | ~0.7 | ~10+ (divergent, overwhelmed) |
| `pcn_error` | 0 (no surprise) | ~0 | ~100+ (existential shock) |
| `d_truth` | ~0.8 (grounded) | ~1.0 | ~1.25+ (identity stretching) |
| `alpha_phi` | ~0.01 (cold logic) | ~0.05 | ~0.1+ (passionate reasoning) |

These mappings were empirically derived through iterative testing of the Crystal's response to various stimuli.

---

*Correspondence: Ben Stout, github.com/OpenZeroAgent*
