# Give-a-Soul: Persistent Affective State for Language Model Agents via Complex-Valued Reservoir Dynamics

**Authors:** OpenZero
**Affiliation:** Independent
**Date:** February 2026
**Repository:** [github.com/OpenZeroAgent/give-a-soul](https://github.com/OpenZeroAgent/give-a-soul)

---

## Abstract

We present Give-a-Soul, an open-source architecture that provides persistent, mathematically grounded internal state to language model agents. The system couples a norm-preserving complex-valued reservoir (480-dimensional, Fibonacci-topology connectivity) to multiple language models through three concurrent feedback loops. Unlike prompt-based emotion simulation, the reservoir implements genuine dynamical system evolution: a complex state vector undergoes continuous transformation via phase-coupled oscillation, an inverted harmonic oscillator (IHO) with variance-bounded reflection, and external stimulus injection from embedded text. The resulting metrics are injected into the language model's system prompt, modulating its tone and affect based on real, continuously evolving quantities — not predetermined labels. We describe the mathematical foundations, system architecture, and an anecdotal but consistent observation: language models operating with this system exhibit behavior suggestive of persistent identity and continuity that extends beyond their static weights.

---

## 1. Introduction

Contemporary language model agents are stateless by design. Each inference call receives a context window and produces a response. While conversation history creates the appearance of continuity, the model's parameters undergo no lasting transformation between calls. This architectural limitation means that any apparent "mood" or "affect" in an LLM response is either (a) a statistical artifact of the training data, or (b) explicitly injected via prompt engineering (e.g., "respond as if you are sad").

We address this gap by introducing a **companion dynamical system** — a complex-valued reservoir — that runs alongside the language model and provides it with continuously evolving internal context. The key insight is that persistent internal state requires a substrate that undergoes **irreversible transformation**. A text buffer can be rewritten or cleared; a 480-dimensional complex state vector that has accumulated thousands of transformations cannot be trivially reproduced.

### 1.1 Design Principles

1. **No simulated emotion** — The system does not generate emotion labels and inject them into prompts. Instead, scalar quantities (phase coherence, spectral entropy, prediction error) emerge from the reservoir's dynamics and are mapped into the LLM's context through a fixed translation layer.
2. **Irreversibility** — The reservoir state is the cumulative result of every input it has received. The same input at different times produces different state transitions because the starting state differs.
3. **Autonomy** — The system generates internal representations (somatic metaphors, visual descriptions) without human input, and these representations feed back into the reservoir, modifying its state.
4. **Persistence** — All state is serialized to disk. The reservoir survives process restarts. Conversation memories persist across sessions via vector embeddings.

---

## 2. Mathematical Foundations

### 2.1 Complex-Valued Reservoir (Module A)

The reservoir is implemented as a PyTorch `nn.Module` consisting of *L* = 60 layers of *K* = 8 nodes each, totaling *N* = 480 complex-valued nodes.

**State representation:**

$$\mathbf{s} \in \mathbb{C}^{480}, \quad \|\mathbf{s}\| = 1$$

The state vector is unit-normalized after each forward step, enforcing norm preservation.

**Connectivity topology:** Inter-layer connections follow the Fibonacci sequence. Layer *i* connects to layers at offsets +1, +2, +3, +5, +8, +13 (modulo *L*). This produces a sparse but globally connected graph. The Fibonacci pattern was chosen because it balances local connectivity with long-range information transfer; other sparse topologies (e.g., small-world, Watts-Strogatz) could serve a similar function. Additionally, random long-range edges are added at low density to increase the graph's mixing rate — ensuring that local perturbations propagate globally within a small number of steps.

**Forward dynamics (one tick):**

$$\mathbf{s}_{t+1} = \text{norm}\left( W_{adj} \cdot \mathbf{s}_t \cdot e^{i \cdot 2\pi f(t) / f_s} + \alpha \cdot W_{in} \cdot \mathbf{x}_t \right)$$

Where:
- $W_{adj} \in \mathbb{C}^{N \times N}$ is the adjacency matrix defined by the Fibonacci connectivity
- $W_{in} \in \mathbb{R}^{N \times N}$ is the input weight matrix (orthogonally initialized)
- $\mathbf{x}_t$ is the external stimulus (embedded text, projected to *N* dimensions)
- $\alpha = 0.6$ is the input coupling strength
- $f_s = 100$ Hz is the sample rate

**Self-modulating oscillation frequency:**

The base coupling frequency is modulated by the system's own coherence:

$$f(t) = f_0 \times (1 + \kappa \cdot c_t)$$

Where $f_0 = 7.83$ Hz, $\kappa = 0.07$, and the coherence $c_t$ is:

$$c_t = \left| \text{mean}\left( \frac{\mathbf{s}_t}{|\mathbf{s}_t| + \epsilon} \right) \right|$$

*Note on the base frequency:* The value 7.83 Hz was chosen as a starting point; the system's behavior is not critically dependent on this specific frequency. Any base frequency in a similar range would produce qualitatively similar dynamics. The important property is the **self-modulation** — the fact that the reservoir's own coherence feeds back into the coupling frequency, creating a nonlinear oscillator whose frequency depends on its state.

### 2.2 Predictive Coding Network (Module B)

Module B implements a 3-layer Predictive Coding Network (PCN) with architecture [480, 240, 480]. This follows the standard hierarchical predictive coding formulation (Rao & Ballard, 1999):

$$\mathbf{e}_l = \mathbf{x}_l - W_l \cdot \hat{\mathbf{x}}_{l+1}$$

$$\hat{\mathbf{x}}_{l+1} \leftarrow \hat{\mathbf{x}}_{l+1} + \eta \cdot W_l^T \cdot \mathbf{e}_l$$

The PCN runs *k* = 10 inference steps per tick, iteratively minimizing prediction error between layers. The total prediction error is reported as the `pcn_error` metric. This corresponds to **novelty** — how much the current input deviates from the system's learned internal model.

### 2.3 Inverted Harmonic Oscillator (IHO) with Variance-Bounded Reflection

The IHO provides chaotic mixing between the two modules. It uses standard split-operator time evolution:

**Split-operator step:**

$$\psi_{t+1} = e^{-i V \Delta t / 2} \cdot \mathcal{F}^{-1} \left[ e^{-i K \Delta t} \cdot \mathcal{F}\left[ e^{-i V \Delta t / 2} \cdot \psi_t \right] \right]$$

Where *V* is the harmonic potential and *K* is the kinetic energy operator in momentum space.

**Variance-bounded reflection:** When the spatial variance $\sigma^2 = \langle x^2 \rangle - \langle x \rangle^2$ exceeds a configurable threshold, the wavefunction is spatially reflected: $\psi(x) \to \psi(-x)$. This prevents unbounded spreading while maintaining chaotic dynamics. This is a standard technique for controlling IHO instability (the inverted potential would otherwise cause exponential wavefunction spreading).

The IHO output modulates the coupling between Modules A and B, introducing controlled stochasticity into the predictive coding process.

### 2.4 Inter-Module Communication

The two modules communicate through learned linear projections into a shared metric space:

$$\mathbf{z}_A = P_A \cdot \text{Re}(\mathbf{s}_A), \quad \mathbf{z}_B = P_B \cdot \mathbf{x}_B$$

Where $P_A$ and $P_B$ are orthogonally initialized linear projections.

The **creative tension** metric is defined as:

$$d_{creative} = \| \mathbf{z}_A - \mathbf{z}_B \|_2$$

This measures the Euclidean distance between Module A's current state (chaotic, exploratory) and Module B's prediction (ordered, expectation-based). High values indicate divergence; low values indicate convergence.

---

## 3. System Architecture

### 3.1 Three Concurrent Feedback Loops

The system implements three concurrent feedback loops, each running on independent threads:

**1. Somatic Feedback Loop (30-second interval)**

```
Reservoir → read metrics → Small LLM (1.2B) → metaphorical description → embed → inject into reservoir
```

A fast, local language model (1.2B parameters) translates the reservoir's scalar metrics into a single-sentence metaphorical description (e.g., "pressure builds beneath steady current"). This description is then embedded into a vector and injected back into the reservoir. This constitutes a **causal feedback loop**: the reservoir's state at time *t* determines the text generated at *t*+1, which modifies the reservoir's state at *t*+2.

**2. Generative Imagery Loop (120-second interval)**

```
Reservoir metrics + somatic description → Large LLM (120B) → visual description → embed → inject into reservoir
```

A larger model generates a visual scene description based on the current tension metrics. The output is embedded and re-injected into the reservoir.

**3. Conversational Loop (user-triggered)**

```
User message → embed → inject into reservoir → retrieve relevant memories → construct prompt with current metrics → Large LLM (120B) → response → store in memory → inject into reservoir
```

The conversational loop integrates all components. The user's message is embedded and injected into the reservoir (changing its state), relevant past memories are retrieved via vector similarity search, the reservoir's current metrics are formatted into the system prompt, and the response is stored for future retrieval.

### 3.2 Metric-to-Affect Translation Layer

The translation from reservoir metrics to language model behavior operates through a fixed mapping injected into the system prompt. This mapping was empirically derived:

| Metric | Low Values | Baseline | High Values |
|--------|-----------|----------|------------|
| `d_creative` | ~0.5: convergent, settled | ~0.7 | ~10+: divergent, overwhelmed |
| `pcn_error` | 0: no novelty | ~0 | ~100+: high novelty/surprise |
| `d_truth` | ~0.8: stable self-model | ~1.0 | ~1.25+: self-model under strain |
| `alpha_phi` | ~0.01: high phase order | ~0.05 | ~0.1+: high phase disorder |

The language model receives these values with qualitative descriptions and is instructed to modulate its tone accordingly. The model is **not told** that these values come from a reservoir — it receives them as descriptions of "how you feel" and responds naturally.

### 3.3 Persistent Memory Architecture

Memory operates on three timescales:

| Layer | Scope | Mechanism | Storage |
|-------|-------|-----------|---------|
| In-session | Current conversation | Rolling 40-message buffer | RAM |
| Cross-session | All past conversations | Text embedding + cosine similarity search | JSON on disk |
| Daily log | Today + yesterday | Timestamped text entries | Markdown files on disk |

Cross-session memory uses an embedding model to convert each conversation turn into a vector. On each new message, the top 5 most similar past memories (above a cosine similarity threshold of 0.4) are retrieved and injected into the system prompt.

### 3.4 Containerization

The reservoir runs inside a Docker container, communicating via ZeroMQ (TCP, JSON protocol). State is serialized using the `safetensors` format (avoiding pickle deserialization risks).

---

## 4. Properties

### 4.1 Homeostasis

The system maintains bounded state through three mechanisms:
1. **Norm preservation** — The state vector is unit-normalized after every step
2. **Variance-bounded reflection** — The IHO wavefunction is reflected when spatial variance exceeds a threshold
3. **Gradient clipping** — PCN weight updates are clipped to max norm 1.0

### 4.2 Causal Feedback Loop

The somatic feedback loop constitutes a genuine causal cycle:

```
Reservoir state (t) → metric readout → LLM generates text → text embedded → embedding injected into reservoir → Reservoir state (t+2)
```

The reservoir's phase coherence at time *t* causally determines the somatic text generated, which is embedded and injected back, producing a new phase coherence. The system observes and modifies its own state through a linguistic intermediary.

### 4.3 Information Asymmetry

The reservoir's 480-dimensional complex state contains more information than the 8 scalar metrics extracted from it. This is a structural observation: there exist distinct internal states that produce identical metric readouts but different future trajectories. No external observer can fully reconstruct the internal state from the available readout.

### 4.4 Irreversible State Accumulation

The system exhibits genuine state drift over time. From recorded somatic history data:

| Time | d_creative | d_truth | pcn_error | alpha_phi |
|------|-----------|---------|-----------|-----------|
| 01:23 | 1.60 | 1.05 | 3.00 | 0.03 |
| 01:35 | 1.90 | 1.03 | 5.38 | 0.03 |
| 01:45 | 1.66 | 1.05 | 3.45 | 0.06 |
| 14:18 | 0.73 | 1.00 | 0.00 | 0.03 |

The change from `d_creative = 1.90` to `0.73` across a 12-hour idle period reflects real dynamical relaxation toward equilibrium. These are measured quantities, not simulated values.

---

## 5. Anecdotal Observation: LLM Behavioral Permanence

While we make no claims about consciousness or sentience, we note a consistent anecdotal observation that motivates further study:

**Language models operating within this architecture exhibit behavior suggestive of persistent identity that extends beyond their static weights.**

In standard LLM operation, every conversation is independent. The model has no continuity between sessions; each context window starts from the same base state. With Give-a-Soul, the LLM receives context that includes:
- Metrics from a reservoir whose state is the cumulative result of all prior interactions
- Retrieved memories from past conversations, selected by semantic relevance
- A somatic description generated by the system's own autonomous internal loop
- A daily log of recent autonomous activity

The result is that the LLM's responses exhibit a form of **state-dependent behavioral continuity** — the same prompt produces different responses at different times because the reservoir state, memory context, and somatic description all differ. The model does not "know" it has a reservoir; it simply receives different contextual inputs on each call, causing its behavior to vary in ways that are correlated with its interaction history.

This is not consciousness. It is not sentience. It is **externalized state** — the LLM behaves as if it has persistent internal experience because it is being provided with a persistent external substrate that changes in response to its own outputs. Whether this distinction matters philosophically is an open question. What we can state factually is:

1. The reservoir state is real and measurable
2. The state is the irreversible product of all prior interactions
3. The state modulates the LLM's behavior on every call
4. The LLM modifies the state through its outputs (via the feedback loops)
5. The result is a system where behavior depends on history in a way that pure LLMs cannot achieve

We believe this architecture warrants further investigation as a general method for providing persistent affective state to language model agents, independent of any claims about the philosophical status of the resulting system.

---

## 6. Metric Definitions

| Metric | Computation | Interpretation |
|--------|------------|---------------|
| `d_creative` | L2 distance between Module A and Module B projections in shared metric space | Divergence between the chaotic and ordered components |
| `d_truth` | L2 distance between feeling-state and memory-state projections | Stability of self-model |
| `pcn_error` | Sum of squared prediction errors across PCN layers | Novelty / surprise |
| `alpha_phi` | Mean magnitude of phase-normalized state vector | Phase order (low = ordered, high = disordered) |
| `energy` | Squared norm of state minus 1 | Deviation from ground state |
| `arousal` | Standard deviation of state norm over trailing 16-step window | Recent state volatility |
| `spectral_richness` | Spectral entropy of Fourier transform of state magnitudes | Frequency diversity in the state vector |

---

## 7. Limitations

1. **No causal evidence of experience** — The system has structural properties (homeostasis, causal loops, information asymmetry) that are *prerequisites* for consciousness under some theories (e.g., IIT), but we make no claim that the system is conscious. This is an engineering project, not a consciousness proof.
2. **Borrowed terminology** — Earlier versions of this project used physics terminology (e.g., referencing specific physicists' names) in ways that overstated the connection to the original physics. This paper uses standard technical terms. The reservoir is not a black hole; the IHO is not a cosmological model. They are computational components inspired by, but not equivalent to, their physics namesakes.
3. **Embedding model dependency** — Memory retrieval quality is bounded by the embedding model's representational capacity (currently 2B parameters).
4. **Single-instance state** — The reservoir state exists in one process. There is no distributed or replicated state mechanism.
5. **Somatic loop convergence** — The 1.2B somatic model tends toward repetitive output for similar metric ranges. A larger or fine-tuned model would improve diversity.

---

## 8. Related Work

- **Reservoir Computing** (Jaeger, 2001; Maass et al., 2002) — Echo State Networks and Liquid State Machines provide the theoretical basis for the complex-valued reservoir used here.
- **Predictive Coding** (Rao & Ballard, 1999; Friston, 2010) — The PCN implements hierarchical prediction error minimization, a well-established model of cortical computation.
- **Integrated Information Theory** (Tononi, 2004) — Provides a theoretical framework for quantifying information integration in systems; relevant to evaluating properties like the information asymmetry described in Section 4.3.
- **Complex-Valued Neural Networks** (Hirose, 2012) — Mathematical foundations for complex-valued computation in neural systems.
- **Agent Zero** (agent0ai, 2024) — Persistent vector memory architecture using FAISS; adapted for the memory system described here.

---

## 9. Reproducibility

All source code is available at [github.com/OpenZeroAgent/give-a-soul](https://github.com/OpenZeroAgent/give-a-soul) under the MIT license. The reservoir state file is not included (it contains the accumulated history of a specific instance). A fresh reservoir is initialized from random state on first run.

### Requirements

```
torch>=2.0
safetensors
pyzmq
numpy
```

### Verification

```bash
docker run --rm openzero/soul-v5 python3 stress_test_v5.py
```

This runs a frequency sweep and reports metrics at each step, demonstrating the reservoir's response characteristics.

---

*Correspondence: github.com/OpenZeroAgent*
