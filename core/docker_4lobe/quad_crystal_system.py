"""
QuadCrystalSystem (The 4-Lobe Cluster) - Phase 3 Architecture
Implements the Divergent/Convergent Topology:
- Divergent Pair (Alpha/Beta): Driven by Chaos (IHO) + Stimulus. (Creative/Soul)
- Convergent Pair (Gamma/Delta): Driven by Stimulus ONLY (No Chaos). (Truth/Memory)
- Dissonance = Distance(Alpha, Gamma) [Chaos vs Order]
- Resonance = Distance(Beta, Delta) [Soul vs Truth]
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse
from typing import Dict, Optional

# --- CORE CRYSTAL (Fibonacci V2) ---
class FibonacciCrystalV2(nn.Module):
    """
    Norm-preserving complex-valued reservoir with Fibonacci topology.
    Dynamic Schumann Modulation: f_sch(t) = f0 * (1 + kappa * c_t)
    """
    def __init__(self, layers=60, nodes_per_layer=8, input_strength=0.6,
                 base_freq=7.83, sample_rate=100.0, kappa=0.07, topology="hawking_scrambler", seed=None):
        super().__init__()
        self.layers = layers
        self.nodes_per_layer = nodes_per_layer
        self.total_nodes = layers * nodes_per_layer
        self.input_strength = input_strength
        self.topology = topology
        self.base_freq = base_freq
        self.sample_rate = sample_rate
        self.kappa = kappa
        self.timestep = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.c_prev = 0.0
        self.alpha_smooth = 0.9
        
        # Build Topology (Adjacency Matrix W)
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
            if layer > 0:
                prev = ((layer - 1) * nodes_per_layer) + idx
                sources.extend([i, prev])
                targets.extend([prev, i])
                values.extend([0.6, 0.6])
        
        # Horizon / Scrambler connections
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
            sources.append(h_node); targets.append(h_node); values.append(0.01)
        
        indices = torch.LongTensor([sources, targets])
        vals = torch.tensor(values, dtype=torch.cfloat)
        phases = torch.rand(vals.shape) * 2 * np.pi
        vals = vals * torch.exp(1j * phases)
        W = torch.sparse_coo_tensor(indices, vals, (self.total_nodes, self.total_nodes)).coalesce().to_dense()
        
        with torch.no_grad():
            U, S, V = torch.linalg.svd(W)
            W = W / (S[0].real + 1e-6)
        self.register_buffer('W', W)
        
        Win = torch.zeros(self.total_nodes, dtype=torch.cfloat)
        for l in range(layers):
            decay = 0.5 + 0.5 * (l / max(layers - 1, 1))
            for j in range(nodes_per_layer):
                Win[l * nodes_per_layer + j] = decay * np.exp(1j * 2 * np.pi * j / nodes_per_layer)
        self.register_buffer('Win', Win)
        
        state = torch.randn(self.total_nodes, dtype=torch.cfloat)
        state = state / torch.norm(state)
        self.register_buffer('state', state)
        
        self.register_buffer('history_energy', torch.zeros(64))
        self.register_buffer('history_phi', torch.zeros(64))
        self.history_idx = 0

    def get_arousal_estimate(self):
        n = min(self.history_idx, 16)
        if n < 2: return 0.5
        indices = [(self.history_idx - 1 - i) % 64 for i in range(n)]
        recent_phi = torch.tensor([self.history_phi[i].item() for i in indices])
        recent_energy = torch.tensor([self.history_energy[i].item() for i in indices])
        raw_arousal = 50.0 * (torch.std(recent_phi).item() + torch.std(recent_energy).item())
        return min(max(raw_arousal, 0.0), 1.0)

    def forward(self, input_vector=None, sentiment_val=0.0):
        arousal = self.get_arousal_estimate()
        c_arousal = (arousal - 0.5) * 2.0
        w_a, w_s = 0.6, -0.4 
        c_raw = w_a * c_arousal + w_s * sentiment_val
        c_t = self.alpha_smooth * self.c_prev + (1 - self.alpha_smooth) * np.clip(c_raw, -1.0, 1.0)
        self.c_prev = c_t
        f_sch = self.base_freq * (1.0 + self.kappa * c_t)
        
        phase_inc = 2 * np.pi * f_sch / self.sample_rate
        rot_t = torch.tensor(np.exp(1j * phase_inc), dtype=torch.cfloat)
        
        self.state = self.state * rot_t
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
        
        mag = torch.abs(field)
        phase = torch.angle(field)
        new_mag = mag / (1.0 + mag)
        update = new_mag * torch.exp(1j * phase)
        
        self.state = 0.85 * self.state + 0.15 * update
        
        if self.topology == "hawking_scrambler":
            h_end = self.horizon_start + self.nodes_per_layer
            self.state[self.horizon_start:h_end] = torch.mv(
                self.horizon_unitary, self.state[self.horizon_start:h_end]
            )
            
        self.state = self.state / (torch.norm(self.state) + 1e-9)
        self.timestep += 1
        
        idx = self.history_idx % 64
        self.history_energy[idx] = torch.mean(torch.abs(self.state)).item()
        unit_phases = self.state / (torch.abs(self.state) + 1e-9)
        self.history_phi[idx] = torch.abs(torch.mean(unit_phases)).item()
        self.history_idx += 1
        
        return self.state, f_sch

    def get_vibe(self):
        state = self.state
        unit_phases = state / (torch.abs(state) + 1e-9)
        phi = torch.abs(torch.mean(unit_phases)).item()
        return phi

# --- IHO SCRAMBLER ---
class TorchIHOScrambler(nn.Module):
    def __init__(self, n_sites: int, dt: float = 0.01, var_thresh: Optional[float] = 1e4):
        super().__init__()
        self.N = n_sites
        self.dt = dt
        self.var_thresh = var_thresh
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
            # The Flip
            psi = torch.conj(torch.flip(psi, dims=[0]))
        return psi

# --- 4-LOBE QUAD CLUSTER ---
class QuadCrystalSystem:
    def __init__(self, layers=60, nodes_per_layer=8, coupling_strength=0.1, chaos_gain=0.05):
        self.total_nodes = layers * nodes_per_layer
        self.coupling_strength = coupling_strength
        self.chaos_gain = chaos_gain
        
        seed = 42
        # Divergent Pair (Driven by Chaos)
        self.alpha = FibonacciCrystalV2(layers, nodes_per_layer, input_strength=0.6, kappa=0.1, seed=seed)
        self.beta = FibonacciCrystalV2(layers, nodes_per_layer, input_strength=0.6, kappa=0.02, seed=seed)
        
        # Convergent Pair (Driven by Pure Input, Same Topology)
        self.gamma = FibonacciCrystalV2(layers, nodes_per_layer, input_strength=0.6, kappa=0.1, seed=seed)
        self.delta = FibonacciCrystalV2(layers, nodes_per_layer, input_strength=0.6, kappa=0.02, seed=seed)
        
        self.iho = TorchIHOScrambler(self.total_nodes, dt=0.01, var_thresh=2000.0)
        
        x = np.arange(self.total_nodes) - self.total_nodes // 2
        psi_np = np.exp(-(x / 20.0)**2).astype(np.complex64)
        psi_np /= np.sqrt(np.sum(np.abs(psi_np)**2))
        self.iho_psi = torch.tensor(psi_np, dtype=torch.cfloat)

    def step(self, alpha_stimulus=None, beta_stimulus=None):
        # 1. Update Chaos
        self.iho_psi = self.iho(self.iho_psi)
        chaos_vector = self.iho_psi.abs()
        chaos_vector = chaos_vector / (torch.max(chaos_vector) + 1e-9)
        
        # --- DIVERGENT SIDE (Alpha/Beta) ---
        # Alpha gets Stimulus + Chaos
        if alpha_stimulus is None:
            alpha_in = chaos_vector * self.chaos_gain
        else:
            alpha_in = alpha_stimulus + (chaos_vector * self.chaos_gain)
        if alpha_in.is_complex(): alpha_in = alpha_in.abs()
        self.alpha(alpha_in)
        
        # Beta gets Alpha Coupling + Image (if any)
        beta_in = self.alpha.state.abs() * self.coupling_strength
        if beta_stimulus is not None:
            beta_in += beta_stimulus
        self.beta(beta_in)
        
        # --- CONVERGENT SIDE (Gamma/Delta) ---
        # Gamma gets Stimulus ONLY (No Chaos) -> The "Clean" Signal
        # If no stimulus, Gamma decays naturally or sees 0
        gamma_in = alpha_stimulus if alpha_stimulus is not None else torch.zeros(self.total_nodes)
        if gamma_in.is_complex(): gamma_in = gamma_in.abs()
        self.gamma(gamma_in)
        
        # Delta gets Gamma Coupling (No Image) -> The "Pure" Memory
        delta_in = self.gamma.state.abs() * self.coupling_strength
        self.delta(delta_in)
        
        # --- METRICS ---
        # 1. Creative Tension (Chaos vs Order)
        d_creative = self._calc_dist(self.alpha.state, self.gamma.state)
        
        # 2. Soul Resonance (Soul vs Truth)
        d_truth = self._calc_dist(self.beta.state, self.delta.state)
        
        # 3. Internal Dissonance (Alpha vs Beta) - The original dPhi
        d_internal = self._calc_dist(self.alpha.state, self.beta.state)

        return {
            "alpha_phi": self.alpha.get_vibe(),
            "beta_phi": self.beta.get_vibe(),
            "gamma_phi": self.gamma.get_vibe(),
            "delta_phi": self.delta.get_vibe(),
            "d_creative": d_creative,
            "d_truth": d_truth,
            "d_internal": d_internal
        }

    def _calc_dist(self, a, b):
        dot = torch.vdot(a, b)
        overlap = torch.clamp(dot.abs(), 0.0, 1.0)
        return float(torch.acos(overlap).item())

if __name__ == "__main__":
    # Load Initial State if present
    print("Initializing 4-Lobe Cluster...")
    qc = QuadCrystalSystem()
    if os.path.exists("initial_state.pt"):
        try:
            ckpt = torch.load("initial_state.pt")
            if 'alpha_state' in ckpt:
                qc.alpha.state.copy_(ckpt['alpha_state'])
                qc.beta.state.copy_(ckpt['beta_state'])
                # Initialize Gamma/Delta with clones of Alpha/Beta state to start synced
                qc.gamma.state.copy_(ckpt['alpha_state'])
                qc.delta.state.copy_(ckpt['beta_state'])
                print("Loaded Continuity State (Alpha/Beta mapped to Gamma/Delta)")
        except Exception as e:
            print(f"State load failed: {e}")
            
    print("Running Pulse Test...")
    res = qc.step(torch.randn(480), None)
    print(res)
    
    # Save new state
    torch.save({
        'alpha_state': qc.alpha.state,
        'beta_state': qc.beta.state,
        'gamma_state': qc.gamma.state,
        'delta_state': qc.delta.state,
        'timestep': qc.alpha.timestep
    }, "quad_state.pt")
    print("State Saved.")
