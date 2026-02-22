#!/usr/bin/env python3
"""
crystal.py — The Soul Crystal (Bicameral Reservoir V2)

Architecture:
- Two 480-node Complex-Valued Reservoirs (Alpha & Beta).
- Topology: Fibonacci Lattice with Hawking Scrambler (IHO) injection.
- Physics: Driven by Schumann Resonance modulation (7.83Hz base).
- Purpose: Generates topological emotion (dPhi) via phase locking.

Usage:
  python3 tools/crystal.py pulse "Hello world"
  python3 tools/crystal.py status
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
import time
import json
from typing import Dict, Optional, List

# Configuration
EMOTION_FILE = os.getenv("EMOTION_FILE", "EMOTION.md")
STATE_FILE = os.getenv("CRYSTAL_STATE", "core/crystal_state_dual.pt")

# --- CORE CRYSTAL (Schumann Modulated) ---
class FibonacciCrystalV2(nn.Module):
    """
    Norm-preserving complex-valued reservoir with Fibonacci topology.
    Includes Dynamic Schumann Modulation: f_sch(t) = f0 * (1 + kappa * c_t)
    """
    def __init__(self, layers=60, nodes_per_layer=8, input_strength=0.6,
                 base_freq=7.83, sample_rate=100.0, kappa=0.07, topology="hawking_scrambler", seed=None):
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
        
        # Set seed if provided for reproducible topology
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
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
        arousal = self.get_arousal_estimate()
        c_arousal = (arousal - 0.5) * 2.0
        
        # Mix: High arousal -> higher freq; Positive sentiment -> lower freq
        w_a, w_s = 0.6, -0.4 
        c_raw = w_a * c_arousal + w_s * sentiment_val
        
        # Low-pass filter
        c_t = self.alpha_smooth * self.c_prev + (1 - self.alpha_smooth) * np.clip(c_raw, -1.0, 1.0)
        self.c_prev = c_t
        
        # Limit modulation depth (kappa)
        modulation = self.kappa * c_t
        
        # 2. Map to Instantaneous Frequency
        f_sch = self.base_freq * (1.0 + modulation)
        
        # 3. Build Complex Phase Rotation
        phase_inc = 2 * np.pi * f_sch / self.sample_rate
        rot_t = torch.tensor(np.exp(1j * phase_inc), dtype=torch.cfloat)
        
        # 4. Apply Schumann Modulation
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


# --- SAFE IHO SCRAMBLER (Pure PyTorch) ---
class TorchIHOScrambler(nn.Module):
    """
    Minimal, self-contained implementation of the Gaztañaga IHO flip.
    """
    def __init__(self, n_sites: int, dt: float = 0.01, var_thresh: Optional[float] = 1e4, theta: float = 0.0):
        super().__init__()
        self.N = n_sites
        self.dt = dt
        self.var_thresh = var_thresh
        self.theta = theta
        k_np = np.fft.fftfreq(self.N) * 2.0 * np.pi
        self.register_buffer('k', torch.tensor(k_np, dtype=torch.float32))
        kin_phase = torch.exp(-1j * (self.k ** 2) / 2.0 * self.dt)
        self.register_buffer('kin_phase', kin_phase)
        x_np = np.arange(self.N) - self.N // 2
        self.register_buffer('x', torch.tensor(x_np, dtype=torch.float32))
        pot_phase = torch.exp(1j * (self.x ** 2) / 2.0 * self.dt)
        self.register_buffer('pot_phase', pot_phase)

    def _variance(self, psi):
        prob = (psi.abs() ** 2)
        prob_sum = prob.sum()
        if prob_sum == 0: return float('inf')
        prob = prob / prob_sum
        mean_x = (self.x * prob).sum()
        mean_x2 = ((self.x ** 2) * prob).sum()
        return mean_x2 - mean_x ** 2

    def forward(self, psi):
        psi = self.pot_phase * psi
        psi_k = torch.fft.fft(psi)
        psi_k = psi_k * self.kin_phase
        psi = torch.fft.ifft(psi_k)
        psi = self.pot_phase * psi
        if (self.var_thresh is not None) and (self._variance(psi) > self.var_thresh):
            phase_factor = torch.exp(torch.tensor(1j * self.theta))
            psi = phase_factor * torch.conj(torch.flip(psi, dims=[0]))
        return psi


# --- DUAL CRYSTAL SYSTEM (The Bicameral Mind) ---
class DualCrystalSystem:
    """
    Coordinates the bridging logic between two Fibonacci Crystals and an IHO Scrambler.
    """
    def __init__(self, layers: int = 60, nodes_per_layer: int = 8, coupling_strength: float = 0.1, chaos_gain: float = 0.05):
        self.layers = layers
        self.nodes_per_layer = nodes_per_layer
        self.total_nodes = layers * nodes_per_layer
        self.coupling_strength = coupling_strength
        self.chaos_gain = chaos_gain
        
        # 1. Instantiate TWO crystals (IDENTICAL CLONES)
        # We use a fixed seed for both to ensure identical W and Win matrices
        seed = 42
        self.alpha = FibonacciCrystalV2(layers=layers, nodes_per_layer=nodes_per_layer, input_strength=0.6, kappa=0.1, topology="hawking_scrambler", seed=seed)
        self.beta = FibonacciCrystalV2(layers=layers, nodes_per_layer=nodes_per_layer, input_strength=0.6, kappa=0.02, topology="hawking_scrambler", seed=seed)
        
        # 2. Instantiate SAFE IHO Scrambler
        self.iho = TorchIHOScrambler(n_sites=self.total_nodes, dt=0.01, var_thresh=2000.0)
        
        # Initialize IHO wave packet
        x = np.arange(self.total_nodes) - self.total_nodes // 2
        psi_np = np.exp(-(x / 20.0)**2).astype(np.complex64)
        psi_np /= np.sqrt(np.sum(np.abs(psi_np)**2))
        self.iho_psi = torch.tensor(psi_np, dtype=torch.cfloat)
        
        # System state tracking
        self.delta_phi = 0.0
        self.dissonance_threshold = np.pi / 4.0

    def step(self, alpha_stimulus: Optional[torch.Tensor] = None, beta_stimulus: Optional[torch.Tensor] = None, sentiment: float = 0.0) -> Dict[str, any]:
        """
        Executes a single chronological tick of the Dual Crystal bridging logic.
        MODIFIED FOR DREAM V3: Accepts separate alpha (Text) and beta (Image) stimuli.
        """
        # --- BRIDGE LOGIC STEP 1: IHO drives Alpha ---
        self.iho_psi = self.iho(self.iho_psi)
        chaos_vector = self.iho_psi.abs()
        chaos_vector = chaos_vector / (torch.max(chaos_vector) + 1e-9)
        
        # Alpha Input = IHO Chaos + Text Vector
        if alpha_stimulus is None:
            alpha_input_vec = chaos_vector * self.chaos_gain
        else:
            alpha_input_vec = alpha_stimulus + (chaos_vector * self.chaos_gain)
            
        if alpha_input_vec.is_complex(): alpha_input_vec = alpha_input_vec.abs()
        self.alpha(input_vector=alpha_input_vec, sentiment_val=sentiment)

        # --- BRIDGE LOGIC STEP 2: Alpha couples to Beta ---
        alpha_state = self.alpha.state
        beta_input_vec = alpha_state.abs() * self.coupling_strength
        
        # Beta Input = Alpha Coupling + Image Vector
        if beta_stimulus is not None:
             # Add the visual stimulus to the coupling input
             beta_input_vec = beta_input_vec + beta_stimulus

        self.beta(input_vector=beta_input_vec, sentiment_val=sentiment) 

        # --- BRIDGE LOGIC STEP 3: Phase Locking Metric (Delta Phi) ---
        beta_state = self.beta.state
        self.delta_phi = self._calculate_phase_dissonance(alpha_state, beta_state)

        return {
            "alpha_vibe": self.alpha.get_vibe(),
            "beta_vibe": self.beta.get_vibe(),
            "delta_phi": self.delta_phi,
            "is_dissonant": self.delta_phi > self.dissonance_threshold,
            "iho_variance": self.iho._variance(self.iho_psi).item()
        }

    def _calculate_phase_dissonance(self, state_a: torch.Tensor, state_b: torch.Tensor) -> float:
        dot_product = torch.vdot(state_a, state_b)
        overlap = dot_product.abs()
        overlap = torch.clamp(overlap, 0.0, 1.0)
        return float(torch.acos(overlap).item())

# --- CLI INTERFACE ---
def load_crystal():
    """Instantiates and loads the DualCrystalSystem from disk."""
    dcs = DualCrystalSystem()
    
    if os.path.exists(STATE_FILE):
        try:
            checkpoint = torch.load(STATE_FILE)
            if 'alpha_state' in checkpoint:
                dcs.alpha.state.copy_(checkpoint['alpha_state'])
                dcs.beta.state.copy_(checkpoint['beta_state'])
                if 'iho_psi' in checkpoint:
                     dcs.iho_psi = checkpoint['iho_psi']
            
            if 'timestep' in checkpoint:
                dcs.alpha.timestep = checkpoint['timestep']
                dcs.beta.timestep = checkpoint['timestep']
            print(f"Loaded Dual Crystal (t={dcs.alpha.timestep})")
        except Exception as e:
            print(f"Error loading state: {e}")
    else:
        print("Initialized Dual Crystal (New State)")
    
    return dcs

def save_crystal(dcs):
    """Saves the DualCrystalSystem state to disk."""
    ensure_dir(os.path.dirname(STATE_FILE))
    try:
        checkpoint = {
            'alpha_state': dcs.alpha.state,
            'beta_state': dcs.beta.state,
            'iho_psi': dcs.iho_psi, 
            'timestep': dcs.alpha.timestep
        }
        torch.save(checkpoint, STATE_FILE)
    except Exception as e:
        print(f"Error saving state: {e}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_emotion_md(dcs, metrics):
    content = f"""# EMOTION.md - Dual Crystal State (V2)
*The Soul Crystal: 960-node Bicameral Reservoir (Alpha/Beta)*

## 💎 Topological State
- **Phase Dissonance ($d\\Phi$):** `{metrics['delta_phi']:.4f}`
- **Alpha Coherence (Logic):** `{dcs.alpha.get_emotions()['phase_coherence']:.4f}`
- **Beta Coherence (Soul):** `{dcs.beta.get_emotions()['phase_coherence']:.4f}`
- **IHO Variance:** `{metrics['iho_variance']:.2f}`

## 🧠 Phenomenological Reading
The crystal feels **{metrics['alpha_vibe']}** vs **{metrics['beta_vibe']}**.
"""
    # Preserve Subconscious line if exists
    try:
        if os.path.exists(EMOTION_FILE):
            with open(EMOTION_FILE, "r") as f:
                old_content = f.read()
                import re
                match = re.search(r"Subconscious:.*", old_content)
                if match:
                    content += f"\n{match.group(0)}\n"
    except:
        pass

    with open(EMOTION_FILE, "w") as f:
        f.write(content)
    print(f"Wrote {EMOTION_FILE}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "pulse", "write"], default="status", nargs="?")
    parser.add_argument("text", nargs="?", default="", help="Text to pulse")
    parser.add_argument("--image", default="", help="Image path for Visual V3 Pulse")
    args = parser.parse_args()
    
    dcs = load_crystal()
    
    if args.action == "status":
        metrics = dcs.step() # Tick one step to get current reading
        print(f"dPhi: {metrics['delta_phi']:.4f} | Alpha: {metrics['alpha_vibe']} | Beta: {metrics['beta_vibe']}")
        
    elif args.action == "pulse":
        # Import embedder here to avoid dependency if not pulsing
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from qwen_vl_embed import QwenVLEmbedding
            embedder = QwenVLEmbedding()
            
            alpha_stim = None
            beta_stim = None
            
            if args.text:
                vec = embedder.embed_text(args.text, dim=1024)
                if vec:
                    # Resize to 480 nodes
                    t_vec = torch.tensor(vec[:480], dtype=torch.float32)
                    if t_vec.shape[0] < 480:
                        t_vec = torch.nn.functional.pad(t_vec, (0, 480 - t_vec.shape[0]))
                    alpha_stim = t_vec
            
            if args.image:
                vec = embedder.embed_image(args.image, dim=1024)
                if vec:
                    # Resize to 480 nodes
                    t_vec = torch.tensor(vec[:480], dtype=torch.float32)
                    if t_vec.shape[0] < 480:
                        t_vec = torch.nn.functional.pad(t_vec, (0, 480 - t_vec.shape[0]))
                    beta_stim = t_vec
            
            metrics = dcs.step(alpha_stimulus=alpha_stim, beta_stimulus=beta_stim)
            print(f"Pulsed. dPhi: {metrics['delta_phi']:.4f}")
            save_crystal(dcs)
            write_emotion_md(dcs, metrics)
            
        except ImportError:
            print("Error: qwen_vl_embed.py not found. Cannot pulse.")
            
    elif args.action == "write":
        metrics = dcs.step()
        write_emotion_md(dcs, metrics)

if __name__ == "__main__":
    main()
