"""
Dual Crystal System (Phase 2 - Pure PyTorch)
Architect: Dr. Light
Refined by: Rock (TorchIHOScrambler)
Implemented by: Roll (Generator)

This module implements the dual hypersphere configuration, coupling the Alpha (Active/Logic)
and Beta (Deep/Soul) Fibonacci crystals, driven by the ternary chaos of the IHOScrambler.

CRITICAL UPGRADE: 
Uses a PURE PYTORCH implementation of the IHO Scrambler (Gaztañaga Flip) to avoid OpenMP conflicts.
"""

import torch
import torch.nn as nn
import numpy as np # Minimal usage for constants
import sys
import os
from typing import Dict, Optional

# --- SAFE IHO SCRAMBLER (Pure PyTorch) ---
class TorchIHOScrambler(nn.Module):
    """
    Minimal, self-contained implementation of the Gaztañaga IHO flip
    using pure PyTorch to ensure thread safety with the Crystal.
    """
    def __init__(self,
                 n_sites: int,
                 dt: float = 0.01,
                 var_thresh: Optional[float] = 1e4,
                 theta: float = 0.0):
        super().__init__()
        self.N = n_sites
        self.dt = dt
        self.var_thresh = var_thresh
        self.theta = theta

        # Lattice spacing dx = 1
        # FFT frequencies are ordered: [0, 1, ..., N/2-1, -N/2, ..., -1]
        k_np = np.fft.fftfreq(self.N) * 2.0 * np.pi
        self.register_buffer('k', torch.tensor(k_np, dtype=torch.float32))
        
        # Kinetic phase: exp(-i * k^2/2 * dt)
        kin_phase = torch.exp(-1j * (self.k ** 2) / 2.0 * self.dt)
        self.register_buffer('kin_phase', kin_phase)

        # Potential phase: exp(-i * V * dt) -> V = -x^2/2 -> exp(i * x^2/2 * dt)
        x_np = np.arange(self.N) - self.N // 2
        self.register_buffer('x', torch.tensor(x_np, dtype=torch.float32))
        
        pot_phase = torch.exp(1j * (self.x ** 2) / 2.0 * self.dt)
        self.register_buffer('pot_phase', pot_phase)

    def _variance(self, psi):
        """Return the spatial variance sigma^2 = <x^2> - <x>^2."""
        # Ensure probability is real and normalized
        prob = (psi.abs() ** 2)
        prob_sum = prob.sum()
        if prob_sum == 0:
            return float('inf')

        prob = prob / prob_sum

        mean_x = (self.x * prob).sum()
        mean_x2 = ((self.x ** 2) * prob).sum()
        return mean_x2 - mean_x ** 2

    def forward(self, psi):
        """
        Perform a single time-step (Split-Operator) + Gaztañaga flip.
        """
        # 1. Half-kick (potential)
        psi = self.pot_phase * psi

        # 2. Full drift (kinetic) via FFT
        psi_k = torch.fft.fft(psi)
        psi_k = psi_k * self.kin_phase
        psi = torch.fft.ifft(psi_k)

        # 3. Second half-kick
        psi = self.pot_phase * psi

        # 4. Optional Gaztañaga flip (Discrete Symmetry Bridge)
        if (self.var_thresh is not None) and (self._variance(psi) > self.var_thresh):
            phase_factor = torch.exp(torch.tensor(1j * self.theta))
            # PT Reflection: x -> -x (reverse array) and t -> -t (complex conjugate)
            psi = phase_factor * torch.conj(torch.flip(psi, dims=[0]))

        return psi


# --- DUAL CRYSTAL SYSTEM ---

# Attempt to import the core crystal
try:
    from .soul_crystal_v2 import FibonacciCrystalV2
except ImportError:
    # Fallback if running as script
    try:
        from soul_crystal_v2 import FibonacciCrystalV2
    except ImportError:
        print("Warning: FibonacciCrystalV2 not found. Using Mock.")
        class FibonacciCrystalV2(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.register_buffer('state', torch.randn(480, dtype=torch.cfloat))
            def forward(self, input_vector=None, sentiment_val=0.0): pass
            def get_vibe(self): return "MockVibe"
            def get_emotions(self): return {"arousal": 0.5}


class DualCrystalSystem:
    """
    Coordinates the bridging logic between two Fibonacci Crystals and an IHO Scrambler.
    """

    def __init__(
        self, 
        layers: int = 60, 
        nodes_per_layer: int = 8,
        coupling_strength: float = 0.1, 
        chaos_gain: float = 0.05
    ):
        self.layers = layers
        self.nodes_per_layer = nodes_per_layer
        self.total_nodes = layers * nodes_per_layer
        self.coupling_strength = coupling_strength
        self.chaos_gain = chaos_gain
        
        # 1. Instantiate TWO crystals
        self.alpha = FibonacciCrystalV2(
            layers=layers, nodes_per_layer=nodes_per_layer, 
            input_strength=0.6,
            kappa=0.1, topology="hawking_scrambler"
        )
        self.beta = FibonacciCrystalV2(
            layers=layers, nodes_per_layer=nodes_per_layer, 
            input_strength=0.6,
            kappa=0.02, topology="hawking_scrambler"
        )
        
        # 2. Instantiate SAFE IHO Scrambler (PyTorch)
        self.iho = TorchIHOScrambler(n_sites=self.total_nodes, dt=0.01, var_thresh=2000.0)
        
        # Initialize IHO wave packet (Gaussian centered at 0)
        x = np.arange(self.total_nodes) - self.total_nodes // 2
        psi_np = np.exp(-(x / 20.0)**2).astype(np.complex64)
        psi_np /= np.sqrt(np.sum(np.abs(psi_np)**2))
        self.iho_psi = torch.tensor(psi_np, dtype=torch.cfloat)
        
        # System state tracking
        self.delta_phi = 0.0
        self.dissonance_threshold = np.pi / 4.0  # 45 degrees

    def step(self, external_stimulus: Optional[torch.Tensor] = None, sentiment: float = 0.0) -> Dict[str, any]:
        """
        Executes a single chronological tick of the Dual Crystal bridging logic.
        """
        
        # --- BRIDGE LOGIC STEP 1: IHO drives Alpha ---
        # Evolve the IHO wave packet
        self.iho_psi = self.iho(self.iho_psi)
        
        # Map wave packet magnitude to chaos vector
        chaos_vector = self.iho_psi.abs()
        # Normalize chaos vector to typical input range (0-1) roughly
        chaos_vector = chaos_vector / (torch.max(chaos_vector) + 1e-9)
        
        # Alpha input
        if external_stimulus is None:
            alpha_input_vec = chaos_vector * self.chaos_gain
        else:
            alpha_input_vec = external_stimulus + (chaos_vector * self.chaos_gain)
            
        # Ensure float32 for input (Crystal expects float input which it complexifies)
        if alpha_input_vec.is_complex():
            alpha_input_vec = alpha_input_vec.abs()
            
        # Alpha updates
        self.alpha(input_vector=alpha_input_vec, sentiment_val=sentiment)

        # --- BRIDGE LOGIC STEP 2: Alpha couples to Beta ---
        alpha_state = self.alpha.state
        
        # Beta sees Alpha's state as its "reality"
        beta_input_vec = alpha_state.abs() * self.coupling_strength
        
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
        """
        Calculates the Phase Locking metric (Delta Phi).
        Formula: Delta Phi = arccos( | <a|b> | )
        """
        # Hermitian inner product
        dot_product = torch.vdot(state_a, state_b)
        overlap = dot_product.abs()
        overlap = torch.clamp(overlap, 0.0, 1.0)
        return float(torch.acos(overlap).item())

if __name__ == "__main__":
    print("Initializing PyTorch Dual Crystal System...")
    dcs = DualCrystalSystem()
    print("Initial State Loaded.")
    
    print("\n--- Running 10 Steps ---")
    for i in range(10):
        # Simulate alternating stress
        sent = -0.5 if i % 2 == 0 else 0.5
        metrics = dcs.step(sentiment=sent)
        print(f"Step {i}: dPhi={metrics['delta_phi']:.4f} | IHO_Var={metrics['iho_variance']:.1f} | {metrics['alpha_vibe']}")
    
    print("\nSystem Functional.")
