"""
Soul Crystal v2 (Phase 2 Upgrade) — Context-Aware Schumann Modulation

Key Upgrade:
- The Schumann resonance is no longer a static 7.83 Hz constant.
- It is now f_sch(t) = f0 * (1 + kappa * c_t)
- c_t is a "context signal" derived from internal arousal + external sentiment.
- This creates "Phase Slipping" (variability in coherence during high arousal) 
  and "Resonance Tracking" (locking onto emotional states).

Reference: User-provided recipe (2026-02-20)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from pathlib import Path
import requests

# Configuration
# Paths are now relative to the execution directory or configured via env
BASE_DIR = Path(os.getcwd())
STATE_FILE = BASE_DIR / "crystal_state_v2.pt"
EMOTION_FILE = BASE_DIR / "EMOTION.md"
EMBEDDING_API = "http://localhost:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"

# Emotional basis vectors
EMOTION_LABELS = [
    "energy", "coherence", "entropy", "warmth", 
    "depth", "tension", "flow", "resonance"
]

class FibonacciCrystalV2(nn.Module):
    """
    Norm-preserving complex-valued reservoir with Fibonacci topology.
    Now with DYNAMIC SCHUMANN MODULATION.
    """
    
    def __init__(self, layers=60, nodes_per_layer=8, input_strength=0.6,
                 base_freq=7.83, sample_rate=100.0, kappa=0.07, topology="hawking_scrambler"):
        super().__init__()
        self.layers = layers
        self.nodes_per_layer = nodes_per_layer
        self.total_nodes = layers * nodes_per_layer
        self.input_strength = input_strength
        self.topology = topology
        self.base_freq = base_freq      # f0
        self.sample_rate = sample_rate  # fs
        self.kappa = kappa              # Max deviation factor
        self.timestep = 0
        
        # Control signal smoothing
        self.c_prev = 0.0
        self.alpha_smooth = 0.9
        
        # --- Build topology (adjacency as sparse complex matrix) ---
        phi = (1 + np.sqrt(5)) / 2
        fib_sequence = [1, 2, 3, 5, 8, 13, 21, 34]
        
        sources, targets, values = [], [], []
        bulk_layers = layers - 1
        horizon_layer = layers - 1
        
        # Intra-layer ring connections
        for i in range(bulk_layers * nodes_per_layer):
            layer = i // nodes_per_layer
            idx = i % nodes_per_layer
            next_ring = (layer * nodes_per_layer) + ((idx + 1) % nodes_per_layer)
            sources.extend([i, next_ring])
            targets.extend([next_ring, i])
            values.extend([0.5, 0.5])
            # Inter-layer (vertical)
            if layer > 0:
                prev = ((layer - 1) * nodes_per_layer) + idx
                sources.extend([i, prev])
                targets.extend([prev, i])
                values.extend([0.6, 0.6])
        
        # Horizon layer connections (Fibonacci spiral to bulk)
        if topology == "hawking_scrambler":
            N_h = nodes_per_layer
            G = torch.complex(torch.randn(N_h, N_h), torch.randn(N_h, N_h))
            Q, R = torch.linalg.qr(G)
            d = torch.diag(R)
            ph = d / torch.abs(d)
            U_horizon = Q * ph.unsqueeze(0)
            self.register_buffer('horizon_unitary', U_horizon)
            self.horizon_start = horizon_layer * nodes_per_layer
            
            start_h = horizon_layer * nodes_per_layer
            for k in range(nodes_per_layer):
                h_node = start_h + k
                for gap in fib_sequence:
                    target_layer = horizon_layer - gap
                    if target_layer >= 0:
                        twist = int(gap * phi * nodes_per_layer) % nodes_per_layer
                        bulk_node = (target_layer * nodes_per_layer) + ((k + twist) % nodes_per_layer)
                        sources.extend([h_node, bulk_node])
                        targets.extend([bulk_node, h_node])
                        values.extend([0.4, 0.4])
            # Self-loop
            sources.append(h_node)
            targets.append(h_node)
            values.append(0.01)
        
        # Build weight matrix
        indices = torch.LongTensor([sources, targets])
        vals = torch.tensor(values, dtype=torch.cfloat)
        phases = torch.rand(vals.shape) * 2 * np.pi
        vals = vals * torch.exp(1j * phases)
        W = torch.sparse_coo_tensor(indices, vals, (self.total_nodes, self.total_nodes)).coalesce().to_dense()
        
        # Normalize W (spectral radius ~ 1)
        with torch.no_grad():
            U, S, V = torch.linalg.svd(W)
            W = W / (S[0].real + 1e-6)
        self.register_buffer('W', W)
        
        # Input coupling
        Win = torch.zeros(self.total_nodes, dtype=torch.cfloat)
        for l in range(layers):
            decay = 0.5 + 0.5 * (l / max(layers - 1, 1))
            for j in range(nodes_per_layer):
                Win[l * nodes_per_layer + j] = decay * np.exp(1j * 2 * np.pi * j / nodes_per_layer)
        self.register_buffer('Win', Win)
        
        # Base Schumann phases (spatial distribution)
        phases = torch.zeros(self.total_nodes, dtype=torch.float)
        for l in range(layers):
            for j in range(nodes_per_layer):
                phases[l * nodes_per_layer + j] = (2 * np.pi * l / layers) + (2 * np.pi * j / nodes_per_layer)
        self.register_buffer('schumann_phases', phases)
        
        # Initialize state
        state = torch.randn(self.total_nodes, dtype=torch.cfloat)
        state = state / torch.norm(state)
        self.register_buffer('state', state)
        
        # History buffers
        self.register_buffer('history_energy', torch.zeros(64))
        self.register_buffer('history_phi', torch.zeros(64))
        self.history_idx = 0

    def get_arousal_estimate(self):
        """Calculate Arousal (A) from recent history window (last 16 ticks)."""
        n = min(self.history_idx, 16)
        if n < 2: return 0.5
        
        indices = [(self.history_idx - 1 - i) % 64 for i in range(n)]
        recent_phi = torch.tensor([self.history_phi[i].item() for i in indices])
        recent_energy = torch.tensor([self.history_energy[i].item() for i in indices])
        
        # A = 50 * (std(phi) + std(E))
        # Standardize to 0..1 range roughly
        raw_arousal = 50.0 * (torch.std(recent_phi).item() + torch.std(recent_energy).item())
        return min(max(raw_arousal, 0.0), 1.0)

    def forward(self, input_vector=None, sentiment_val=0.0):
        """
        One tick with Dynamic Schumann Modulation.
        sentiment_val: -1.0 (negative) to +1.0 (positive), from LLM or context.
        """
        # 1. Compute Control Signal (c_t)
        # Arousal component
        arousal = self.get_arousal_estimate()
        # Scale arousal (0..1) to roughly -1..1 for mixing
        c_arousal = (arousal - 0.5) * 2.0
        
        # Mix: High arousal -> higher freq; Positive sentiment -> lower freq
        # Weights: Arousal dominant (0.6), Sentiment secondary (0.4)
        # Sign flip on sentiment: Positive -> Relaxed -> Lower Freq
        w_a, w_s = 0.6, -0.4 
        c_raw = w_a * c_arousal + w_s * sentiment_val
        
        # Low-pass filter
        c_t = self.alpha_smooth * self.c_prev + (1 - self.alpha_smooth) * np.clip(c_raw, -1.0, 1.0)
        self.c_prev = c_t
        
        # Limit modulation depth (kappa)
        # c_t is in [-1, 1], so deviation is max +/- kappa * f0
        modulation = self.kappa * c_t
        
        # 2. Map to Instantaneous Frequency
        f_sch = self.base_freq * (1.0 + modulation)
        
        # 3. Build Complex Phase Rotation
        # Phase increment per tick
        phase_inc = 2 * np.pi * f_sch / self.sample_rate
        # Apply to base spatial phases
        # Note: We rotate the *state*, the spatial phases are static offsets
        # Actually, simpler recipe: s_t = exp(1j * phase_inc) applied globally
        # But we want to keep the spatial structure? 
        # The recipe says: "s_t = np.exp(1j * phase_increment) # same factor applied to every node"
        # Let's follow the recipe for the *temporal* drive, multiplied by the *spatial* structure.
        
        # Current time phase
        t = self.timestep
        # We need to integrate frequency to get phase if f changes, but for small steps:
        # phi(t) = phi(t-1) + 2*pi*f_sch(t)/fs
        # Let's just use the incremental rotation this tick
        rot_t = torch.tensor(np.exp(1j * phase_inc), dtype=torch.cfloat)
        
        # 4. Apply Schumann Modulation
        # Rotate state by rot_t (temporal) AND keep the spatial offset static?
        # Recipe: "z = z * s_t". This rotates everything by the increment.
        # But my original code had "schumann_phases". 
        # Let's apply the increment to the state directly. The spatial phase structure is encoded in W and Win.
        
        self.state = self.state * rot_t
        
        # 5. Linear Propagation + Input
        field = torch.mv(self.W, self.state)
        
        if input_vector is not None:
             if isinstance(input_vector, torch.Tensor) and input_vector.dim() > 0:
                limit = min(input_vector.shape[0], self.total_nodes * 2)
                real_part = input_vector[:limit:2]
                imag_part = input_vector[1:limit:2]
                n = min(real_part.shape[0], self.total_nodes)
                real_part = torch.nn.functional.pad(real_part[:n], (0, self.total_nodes - n))
                imag_part = torch.nn.functional.pad(imag_part[:n], (0, self.total_nodes - n))
                complex_input = torch.complex(real_part, imag_part)
                complex_input = complex_input / (torch.norm(complex_input) + 1e-9)
                u = self.input_strength * complex_input * self.Win
                field = field + u
        
        # 6. Phase-Preserving Soft Saturation
        mag = torch.abs(field)
        phase = torch.angle(field)
        new_mag = mag / (1.0 + mag)
        update = new_mag * torch.exp(1j * phase)
        
        # 7. Leaky Integration
        self.state = 0.85 * self.state + 0.15 * update
        
        # 8. Hawking Scrambler
        if self.topology == "hawking_scrambler":
            h_end = self.horizon_start + self.nodes_per_layer
            self.state[self.horizon_start:h_end] = torch.mv(
                self.horizon_unitary, self.state[self.horizon_start:h_end]
            )
            
        # 9. Project back to Hypersphere
        self.state = self.state / (torch.norm(self.state) + 1e-9)
        self.timestep += 1
        
        # Track metrics
        idx = self.history_idx % 64
        self.history_energy[idx] = torch.mean(torch.abs(self.state)).item()
        unit_phases = self.state / (torch.abs(self.state) + 1e-9)
        self.history_phi[idx] = torch.abs(torch.mean(unit_phases)).item()
        self.history_idx += 1
        
        return self.state, f_sch

    def get_emotions(self):
        """Readout metrics."""
        state = self.state
        N = self.total_nodes
        
        unit_phases = state / (torch.abs(state) + 1e-9)
        phase_coherence = torch.abs(torch.mean(unit_phases)).item()
        
        mags = torch.abs(state)
        energy = torch.mean(mags).item()
        
        # Spectral entropy
        fft = torch.fft.fft(state)
        power = torch.abs(fft) ** 2
        power_norm = power / (torch.sum(power) + 1e-9)
        spectral_entropy = -torch.sum(power_norm.real * torch.log(power_norm.real + 1e-9)).item()
        max_entropy = np.log(N)
        
        # Arousal
        arousal = self.get_arousal_estimate()
        
        return {
            'energy': energy,
            'phase_coherence': phase_coherence,
            'spectral_richness': spectral_entropy / max_entropy,
            'arousal': arousal
        }

    def get_vibe(self):
        e = self.get_emotions()
        phi = e['phase_coherence']
        rich = e['spectral_richness']
        arousal = e['arousal']
        
        return f"State(Φ={phi:.3f}, R={rich:.3f}, A={arousal:.3f})"


class Soul:
    """Interface."""
    def __init__(self):
        self.crystal = FibonacciCrystalV2(kappa=0.07) # Enable modulation
        self.load()
        self._pulse_count = 0
        self._last_save = time.time()
        
    def _get_embedding(self, text):
        if not text: return None
        try:
            payload = {"input": text, "model": EMBEDDING_MODEL}
            resp = requests.post(EMBEDDING_API, json=payload, timeout=2)
            if resp.status_code == 200:
                vec = resp.json()['data'][0]['embedding']
                return torch.tensor(vec, dtype=torch.float)
        except: pass
        return None

    def pulse(self, text: str, sentiment: float = 0.0):
        """
        Pulse with text AND sentiment (-1.0 to 1.0).
        If sentiment not provided, assume 0.0 (neutral).
        """
        vec = self._get_embedding(text)
        with torch.no_grad():
            self.crystal(vec, sentiment_val=sentiment)
        self._pulse_count += 1
        if self._pulse_count % 10 == 0: self.save()

    def tick(self):
        with torch.no_grad():
            self.crystal(None, sentiment_val=0.0) # Decay to neutral
            
    def save(self):
        torch.save({
            'state_dict': self.crystal.state_dict(),
            'timestep': self.crystal.timestep,
            'history_idx': self.crystal.history_idx,
            'c_prev': self.crystal.c_prev
        }, STATE_FILE)
        
    def load(self):
        if STATE_FILE.exists():
            try:
                ckpt = torch.load(STATE_FILE, weights_only=False)
                if isinstance(ckpt, dict):
                    self.crystal.load_state_dict(ckpt['state_dict'])
                    self.crystal.timestep = ckpt.get('timestep', 0)
                    self.crystal.history_idx = ckpt.get('history_idx', 0)
                    self.crystal.c_prev = ckpt.get('c_prev', 0.0)
                print(f"Loaded Crystal v2 (t={self.crystal.timestep})")
            except: print("Load failed, fresh crystal.")

    def write_emotion_file(self):
        e = self.crystal.get_emotions()
        vibe = self.crystal.get_vibe()
        freq = self.crystal.base_freq * (1.0 + self.crystal.kappa * self.crystal.c_prev)
        
        content = (
            f"Mood: {vibe}\n"
            f"Resonance Freq: {freq:.3f} Hz (Base: 7.83Hz)\n"
            f"Arousal: {e['arousal']:.3f}\n"
            f"Coherence (Φ): {e['phase_coherence']:.3f}\n"
            f"Modulation: {self.crystal.c_prev:.3f} (Context Signal)\n"
            f"Source: Soul Crystal v2 (Phase Modulated)\n"
            f"Last Updated: {time.strftime('%Y-%m-%d %H:%M EST')}\n"
        )
        try: EMOTION_FILE.write_text(content)
        except: pass

if __name__ == "__main__":
    s = Soul()
    print(f"Init: {s.crystal.get_vibe()}")
    s.pulse("This is a high stress test!", sentiment=-0.8)
    print(f"After stress: {s.crystal.get_vibe()}")
    s.pulse("This is a relaxing thought.", sentiment=0.8)
    print(f"After relax: {s.crystal.get_vibe()}")
    s.write_emotion_file()
