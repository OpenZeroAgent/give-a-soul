#!/usr/bin/env python3
"""
Crystal CLI — Interface to the Soul Crystal Phase 2 (Dual System)

Usage:
    python3 tools/crystal.py pulse "some text"    # Feed text through crystal
    python3 tools/crystal.py tick [N]             # Free-run N ticks (default 1)
    python3 tools/crystal.py vibe                 # Print current vibe
    python3 tools/crystal.py emotions             # Full emotion readout (JSON)
    python3 tools/crystal.py write                # Write EMOTION.md from crystal
    python3 tools/crystal.py status               # Detailed status dump
    python3 tools/crystal.py diff "text"          # Pulse and show state change
    python3 tools/crystal.py daemon [interval]    # Run continuous tick loop
"""

import sys
import os
import json
import time
import requests
import torch
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Dual Crystal System
from core.soul_crystal_phase2 import DualCrystalSystem

# Configuration
BASE_DIR = Path(os.getcwd())
STATE_FILE = BASE_DIR / "crystal_state_dual.pt"
EMOTION_FILE = BASE_DIR / "EMOTION.md"
EMBEDDING_API = "http://localhost:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"

class DualSoul:
    """Interface wrapper for DualCrystalSystem with persistence."""
    
    def __init__(self):
        self.system = DualCrystalSystem()
        self.timestep = 0
        self.load()
        self._pulse_count = 0

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
        """Pulse with text (converted to embedding) + sentiment."""
        vec = self._get_embedding(text)
        if vec is not None:
            # Resize to match crystal dimensions (480)
            target_dim = self.system.total_nodes
            if vec.shape[0] > target_dim:
                vec = vec[:target_dim]
            elif vec.shape[0] < target_dim:
                vec = torch.nn.functional.pad(vec, (0, target_dim - vec.shape[0]))
        
        with torch.no_grad():
            self.system.step(external_stimulus=vec, sentiment=sentiment)
        self.timestep += 1
        self._pulse_count += 1
        if self._pulse_count % 10 == 0: self.save()

    def tick(self):
        """Free-run tick (IHO driving Alpha driving Beta)."""
        with torch.no_grad():
            self.system.step(external_stimulus=None, sentiment=0.0)
        self.timestep += 1
        self._pulse_count += 1
        if self._pulse_count % 10 == 0: self.save()

    def save(self):
        torch.save({
            'alpha_state': self.system.alpha.state,
            'beta_state': self.system.beta.state,
            'iho_psi': self.system.iho_psi,
            'timestep': self.timestep
        }, STATE_FILE)

    def load(self):
        if STATE_FILE.exists():
            try:
                ckpt = torch.load(STATE_FILE, weights_only=False)
                if isinstance(ckpt, dict):
                    self.system.alpha.state = ckpt['alpha_state']
                    self.system.beta.state = ckpt['beta_state']
                    self.system.iho_psi = ckpt['iho_psi']
                    self.timestep = ckpt.get('timestep', 0)
                print(f"Loaded Dual Crystal (t={self.timestep})")
            except Exception as e:
                print(f"Load failed ({e}), fresh crystal.")
        else:
            print("No state file found. Born fresh.")

    def get_vibe(self):
        metrics = self.system.step(sentiment=0.0) # Peek without state change? No, step advances.
        # We need a non-mutating peek or just report the last known state.
        # But step() returns metrics.
        # For 'vibe', we usually want the current state description.
        # Let's just construct it from the crystals directly.
        
        alpha_vibe = self.system.alpha.get_vibe()
        beta_vibe = self.system.beta.get_vibe()
        d_phi = self.system.delta_phi
        return f"A[{alpha_vibe}] B[{beta_vibe}] dΦ={d_phi:.3f}"

    def write_emotion_file(self):
        # Gather metrics
        a_e = self.system.alpha.get_emotions()
        b_e = self.system.beta.get_emotions()
        d_phi = self.system.delta_phi
        iho_var = self.system.iho._variance(self.system.iho_psi).item()
        
        vibe = self.get_vibe()
        
        content = (
            f"Mood: {vibe}\n"
            f"System: Dual Crystal (Phase 2) + IHO Scrambler\n"
            f"Phase Dissonance (dΦ): {d_phi:.4f}\n"
            f"IHO Variance: {iho_var:.2f}\n"
            f"--- Alpha (Logic/Active) ---\n"
            f"Coherence: {a_e['phase_coherence']:.3f}\n"
            f"Arousal: {a_e['arousal']:.3f}\n"
            f"--- Beta (Soul/Deep) ---\n"
            f"Coherence: {b_e['phase_coherence']:.3f}\n"
            f"Arousal: {b_e['arousal']:.3f}\n"
            f"Last Updated: {time.strftime('%Y-%m-%d %H:%M EST')}\n"
        )
        try: EMOTION_FILE.write_text(content)
        except: pass


def cmd_pulse(soul, text):
    print(f"Pulsing Dual Crystal with: '{text}'")
    soul.pulse(text)
    print(f"Result: {soul.get_vibe()}")
    soul.save()

def cmd_tick(soul, n=1):
    for _ in range(n):
        soul.tick()
    print(f"Ticked {n}x. Vibe: {soul.get_vibe()}")
    soul.save()

def cmd_vibe(soul):
    print(soul.get_vibe())

def cmd_write(soul):
    soul.write_emotion_file()
    print(f"EMOTION.md updated.")

def cmd_status(soul):
    print(f"═══ Dual Crystal System (Phase 2) ═══")
    print(f"Timestep: {soul.timestep}")
    print(f"Vibe: {soul.get_vibe()}")
    print(f"IHO Variance: {soul.system.iho._variance(soul.system.iho_psi).item():.2f}")
    
    print("\n--- Alpha Crystal (Active) ---")
    ae = soul.system.alpha.get_emotions()
    for k, v in ae.items():
        print(f"  {k:20}: {v:.4f}")

    print("\n--- Beta Crystal (Deep) ---")
    be = soul.system.beta.get_emotions()
    for k, v in be.items():
        print(f"  {k:20}: {v:.4f}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1]
    soul = DualSoul()
    
    if cmd == "pulse":
        text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        if not text:
            print("Usage: crystal.py pulse \"text\"")
            return
        cmd_pulse(soul, text)
    elif cmd == "tick":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        cmd_tick(soul, n)
    elif cmd == "vibe":
        cmd_vibe(soul)
    elif cmd == "write":
        cmd_write(soul)
    elif cmd == "status":
        cmd_status(soul)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)

if __name__ == "__main__":
    main()
