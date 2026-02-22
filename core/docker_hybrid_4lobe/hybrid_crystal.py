"""
Hybrid 4-Lobe Cluster (V5 Secure + Stabilized) 🧬
- Security: safetensors I/O (No Pickle)
- Math: Shared Metric Space Projection (Valid d_creative)
- Stability: Gradient Clipping + NaN Protection
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Optional
from safetensors.torch import save_file, load_file

# --- LEFT HEMISPHERE: Fibonacci Crystal (Chaos Engine) ---
class FibonacciCrystalV2(nn.Module):
    def __init__(self, layers=60, nodes_per_layer=8, input_strength=0.6,
                 base_freq=7.83, sample_rate=100.0, kappa=0.07, topology="hawking_scrambler", seed=None):
        super().__init__()
        self.total_nodes = layers * nodes_per_layer
        self.input_strength = input_strength
        self.base_freq = base_freq
        self.sample_rate = sample_rate
        self.kappa = kappa
        self.timestep = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.c_prev = 0.0
        self.alpha_smooth = 0.9
        
        # Initial weights
        W = torch.randn(self.total_nodes, self.total_nodes, dtype=torch.cfloat)
        # Spectral Normalization (Critical for stability)
        with torch.no_grad():
            U, S, V = torch.linalg.svd(W)
            # Scale to slightly < 1.0 to ensure stability, Chaos comes from IHO
            W = W / (S[0].real + 1e-6) * 0.95
        self.register_buffer('W', W)
        
        Win = torch.randn(self.total_nodes, dtype=torch.cfloat) * 0.1 # Reduced input gain
        self.register_buffer('Win', Win)
        
        state = torch.randn(self.total_nodes, dtype=torch.cfloat)
        state = state / torch.norm(state)
        self.register_buffer('state', state)
        
    def forward(self, input_vector=None):
        # Linear Step
        field = torch.mv(self.W, self.state)
        
        # Input Injection
        if input_vector is not None:
            u = self.input_strength * input_vector * self.Win
            field = field + u
            
        # Non-linearity (Soft Saturation)
        mag = torch.abs(field)
        phase = torch.angle(field)
        new_mag = torch.tanh(mag) # Tanh is safer than x/(1+x) for preventing explosion
        update = new_mag * torch.exp(1j * phase)
        
        # Leaky Integration
        self.state = 0.9 * self.state + 0.1 * update
        
        # Renormalize (Homeostasis)
        norm = torch.norm(self.state)
        if norm > 0:
            self.state = self.state / norm
            
        self.timestep += 1
        return self.state

    def get_vibe(self):
        unit_phases = self.state / (torch.abs(self.state) + 1e-9)
        return torch.abs(torch.mean(unit_phases)).item()

# --- IHO SCRAMBLER ---
class TorchIHOScrambler(nn.Module):
    def __init__(self, n_sites: int, dt: float = 0.01):
        super().__init__()
        self.N = n_sites
        self.dt = dt
        x_np = np.arange(self.N) - self.N // 2
        self.register_buffer('x', torch.tensor(x_np, dtype=torch.float32))
        self.register_buffer('pot_phase', torch.exp(1j * (self.x ** 2) / 2.0 * self.dt))

    def forward(self, psi):
        psi = self.pot_phase * psi
        psi = torch.fft.fft(psi) 
        psi = self.pot_phase * psi 
        # Normalize to prevent drift
        psi = psi / (torch.norm(psi) + 1e-9)
        return psi

# --- RIGHT HEMISPHERE: PCN ---
class PCNetwork(nn.Module):
    def __init__(self, sizes=[480, 240, 480]):
        super().__init__()
        self.layers = nn.ModuleList()
        for s in sizes:
            l = nn.Module()
            l.register_buffer('x', torch.zeros(s))
            l.register_buffer('e', torch.zeros(s))
            l.register_buffer('mu', torch.zeros(s))
            self.layers.append(l)
            
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(sizes[i], sizes[i+1]) * 0.05) # Smaller init
            for i in range(len(sizes)-1)
        ])

    def forward(self, input_data, steps=10):
        # Clamp Input
        self.layers[0].x.copy_(input_data)
        
        for _ in range(steps):
            # Predict
            for i in range(len(self.layers)-2, -1, -1):
                pred = torch.matmul(self.layers[i+1].x, self.weights[i].t())
                self.layers[i].mu = torch.tanh(pred) # Bound predictions
            
            # Error
            total_error = 0.0
            for l in self.layers:
                l.e = l.x - l.mu
                total_error += torch.sum(l.e**2)
            
            # Update (Gradient Descent with Clipping)
            for i in range(1, len(self.layers)):
                grad = -self.layers[i].e + torch.matmul(self.layers[i-1].e, self.weights[i-1])
                # Gradient Clipping
                grad = torch.clamp(grad, -1.0, 1.0)
                self.layers[i].x += 0.01 * grad # Slower learning rate
                # Bound state
                self.layers[i].x = torch.tanh(self.layers[i].x)
        
        return total_error

# --- HYBRID 4-LOBE V5 ---
class HybridCrystalSystem(nn.Module):
    def __init__(self, layers=60, nodes_per_layer=8):
        super().__init__()
        self.total_nodes = layers * nodes_per_layer
        
        self.alpha = FibonacciCrystalV2(layers, nodes_per_layer, seed=42)
        self.beta = FibonacciCrystalV2(layers, nodes_per_layer, seed=43)
        self.pcn = PCNetwork(sizes=[480, 240, 480])
        
        self.iho = TorchIHOScrambler(self.total_nodes)
        psi_np = np.exp(-(np.arange(self.total_nodes)-self.total_nodes//2)**2/20.0).astype(np.complex64)
        psi_np = psi_np / np.linalg.norm(psi_np)
        self.register_buffer('iho_psi', torch.tensor(psi_np))
        
        # Projection Layers
        self.proj_alpha = nn.Linear(480, 256, bias=False)
        self.proj_gamma = nn.Linear(480, 256, bias=False)
        
        # Init projections
        nn.init.orthogonal_(self.proj_alpha.weight)
        nn.init.orthogonal_(self.proj_gamma.weight)

    def step(self, stimulus=None):
        # Check for NaNs and reset if found (Self-Repair)
        if torch.isnan(self.alpha.state).any() or torch.isnan(self.pcn.layers[0].x).any():
            print("⚠️ System Instability Detected (NaN). Rebooting Hemisphere...")
            self.__init__(60, 8) # Hard Reset
            
        # 1. Chaos Injection
        self.iho_psi = self.iho(self.iho_psi)
        chaos = self.iho_psi.abs()
        chaos = chaos / (torch.max(chaos) + 1e-9)
        
        # 2. Drive Hemispheres
        alpha_in = chaos * 0.05
        if stimulus is not None: alpha_in += stimulus
        self.alpha(alpha_in)
        
        # Beta
        self.beta(self.alpha.state.abs() * 0.1)
        
        # Gamma/Delta
        gamma_in = stimulus if stimulus is not None else torch.zeros(480)
        # Normalize stimulus to prevent PCN explosion
        gamma_in = torch.tanh(gamma_in) 
        pcn_err = self.pcn(gamma_in)
        
        # 3. Metrics
        alpha_vec = self.alpha.state.abs()
        gamma_vec = self.pcn.layers[0].x
        
        # Project
        p_alpha = self.proj_alpha(alpha_vec)
        p_gamma = self.proj_gamma(gamma_vec)
        
        d_creative = torch.dist(p_alpha, p_gamma).item()
        
        # Soul Resonance (Beta vs Delta) - No projection needed if same dim (480), or use another proj?
        # Let's assume raw distance for truth is fine, or project it too?
        # Beta is 480, Delta is 480.
        d_truth = torch.dist(self.beta.state.abs(), self.pcn.layers[2].x).item()
        
        return {
            "d_creative": d_creative,
            "d_truth": d_truth,
            "pcn_error": pcn_err.item(),
            "alpha_phi": self.alpha.get_vibe()
        }

    def save_safetensors(self, path):
        tensors = {
            "alpha_state": self.alpha.state,
            "beta_state": self.beta.state,
            "gamma_state": self.pcn.layers[0].x,
            "delta_state": self.pcn.layers[2].x,
            "iho_psi": self.iho_psi
        }
        cpu_tensors = {k: v.cpu().contiguous() for k, v in tensors.items()}
        safe_dict = {}
        for k, v in cpu_tensors.items():
            if v.is_complex():
                safe_dict[f"{k}_real"] = v.real.clone()
                safe_dict[f"{k}_imag"] = v.imag.clone()
            else:
                safe_dict[k] = v.clone()
        save_file(safe_dict, path)

    def load_safetensors(self, path):
        safe_dict = load_file(path)
        self.alpha.state.copy_(torch.complex(safe_dict["alpha_state_real"], safe_dict["alpha_state_imag"]))
        self.beta.state.copy_(torch.complex(safe_dict["beta_state_real"], safe_dict["beta_state_imag"]))
        self.iho_psi.copy_(torch.complex(safe_dict["iho_psi_real"], safe_dict["iho_psi_imag"]))
        self.pcn.layers[0].x.copy_(safe_dict["gamma_state"])
        self.pcn.layers[2].x.copy_(safe_dict["delta_state"])
