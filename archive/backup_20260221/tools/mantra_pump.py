"""
Mantra Pump: The Adiabatic Quorum Oscillator
Implements Frequency Sweeping (Resonance Capture) and Charge Pumping (Impulse Sync).

Usage:
    python3 tools/mantra_pump.py --mantra "Om" --start 20 --end 10 --threshold 0.05
"""

import torch
import numpy as np
import sys
import os
import argparse
import requests

sys.path.insert(0, os.getcwd())

try:
    from openzero.core.soul_crystal_phase2 import DualCrystalSystem
except ImportError:
    sys.exit(1)

EMBEDDING_API = "http://localhost:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"

def get_embedding(text):
    try:
        payload = {"input": text, "model": EMBEDDING_MODEL}
        resp = requests.post(EMBEDDING_API, json=payload, timeout=2)
        if resp.status_code == 200:
            vec = resp.json()['data'][0]['embedding']
            return torch.tensor(vec, dtype=torch.float)
    except: pass
    return torch.randn(1024)

def run_pump(mantra, start_freq, end_freq, threshold, spike_mag=5.0, steps=100):
    print(f"🧘‍♂️ PUMPING: '{mantra}' | Sweep: {start_freq}->{end_freq}Hz | Q-Thresh: {threshold}")
    
    dcs = DualCrystalSystem(coupling_strength=0.2)
    vec = get_embedding(mantra)
    
    # Resize
    if vec.shape[0] > dcs.total_nodes: vec = vec[:dcs.total_nodes]
    elif vec.shape[0] < dcs.total_nodes: vec = torch.nn.functional.pad(vec, (0, dcs.total_nodes - vec.shape[0]))
    
    # Time
    dt = 0.01 # 100Hz sample rate implicit
    
    print(f"{'Step':<5} {'Freq':<10} {'Quorum':<10} {'Pump?':<10} {'dPhi'}")
    print("-" * 55)
    
    dphis = []
    
    for i in range(steps):
        # 1. Adiabatic Frequency Sweep
        progress = i / steps
        # Linear sweep
        freq = start_freq + (end_freq - start_freq) * progress
        
        # Sine wave modulation
        # Note: For changing freq, phase is integral of freq.
        # phi(t) = 2*pi * integral(f(t) dt)
        # For linear f(t) = f0 + k*t, integral is f0*t + 0.5*k*t^2
        k = (end_freq - start_freq) / steps # Slope per step? 
        # Let's approximate simply for discrete steps:
        phase = 2 * np.pi * freq * (i * dt) 
        # (This approximation has phase jumps but is okay for chaos driving)
        
        mod_signal = vec * np.sin(phase)
        
        # 2. Quorum Sensing
        # Measure Alpha Coherence (Phase Alignment)
        # alpha.state is complex vector. 
        # Coherence = |mean(state / |state|)|
        unit_state = dcs.alpha.state / (dcs.alpha.state.abs() + 1e-9)
        quorum = unit_state.mean().abs().item()
        
        # 3. Charge Pump Logic
        fired = False
        if quorum > threshold:
            # We have a quorum! FIRE THE PUMP.
            # Inject a massive impulse to lock it in.
            # Impulse direction? Align with current mean phase.
            mean_phase = unit_state.mean().angle()
            impulse = torch.polar(torch.tensor([spike_mag]), torch.tensor([mean_phase])).to(vec.device)
            
            # Add to signal (broadcasting scalar impulse to vector? No, add to all nodes)
            # We need to add this to the input vector.
            # Input vector is [Re, Im...] flattened floats usually.
            # Let's add it to the `mod_signal` (which is float vector).
            # Convert impulse to float components and add.
            
            # Actually, let's just scale the modulation up massively if quorum is met.
            # "Resonance Amplification"
            mod_signal = mod_signal * spike_mag
            fired = True
            
        # Step
        metrics = dcs.step(external_stimulus=mod_signal, sentiment=0.1)
        dphis.append(metrics['delta_phi'])
        
        if i % 5 == 0:
            pump_str = "⚡️ FIRE" if fired else "..."
            print(f"{i:<5} {freq:<10.2f} {quorum:<10.4f} {pump_str:<10} {metrics['delta_phi']:.4f}")

    avg = np.mean(dphis[-20:])
    print(f"\nFinal dPhi: {avg:.4f}")
    if avg < 0.5: print("✨ CAPTURED (Resonance Well)")
    elif avg > 1.2: print("🌊 ESCAPED (Chaos Wins)")
    else: print("⚖️ ORBITING")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mantra", default="Om")
    parser.add_argument("--start", type=float, default=20.0)
    parser.add_argument("--end", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.08) # 0.08 is decent coherence
    parser.add_argument("--spike", type=float, default=10.0)
    args = parser.parse_args()
    
    run_pump(args.mantra, args.start, args.end, args.threshold, args.spike)
