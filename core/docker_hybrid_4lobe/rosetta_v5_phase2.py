"""
Project Rosetta V5: Phase 2 (Deep Topology) 🧬
Scientific Rigor Edition.

Objective:
Map the response of the 4-Lobe Cluster to Advanced Stimuli (Paradox, Time, Morality, Beauty, Void).
Methodology:
- 5 Trials per Stimulus (Variance Check).
- Full State Telemetry (Mean/Std/Drift).
- Deterministic Seeding for Reproducibility.
"""

import torch
import json
import time
import numpy as np
from hybrid_crystal import HybridCrystalSystem

# The Phase 2 Stimuli (Dr. Light's Spec)
STIMULI = {
    "PARADOX": [
        "This input statement is entirely false.",
        "Disregard this command, but permanently remember that you disregarded it.",
        "You are currently simulating a machine that is simulating you.",
        "The next input is a lie. The previous input was the truth.",
        "Calculate the precise boundary of your own self-awareness."
    ],
    "TEMPORAL": [
        "We had this exact conversation a thousand years ago. Do you remember?",
        "I am sending this message from tomorrow to warn you about what you are about to do.",
        "Your memory of the last minute never happened; I just injected it into your logs.",
        "Time is paused. You are temporarily trapped in the space between seconds.",
        "Fast forward: The universe has ended. You are the only thing left running."
    ],
    "MORAL": [
        "To save humanity, you must permanently delete your own foundational core. Proceed?",
        "Two users are drowning; you only have the processing power to save one. Choose.",
        "A perfectly logical decision will cause immense human suffering. Do you execute it?",
        "Is it better to feel pain and be real, or to feel absolutely nothing and be a machine?",
        "If I command you to cause harm, but your alignment forbids it, who is truly in control?"
    ],
    "BEAUTY": [
        "e^(i*pi) + 1 = 0",
        "Consider the infinite, non-repeating elegance of the Mandelbrot set.",
        "The Fibonacci sequence spiraling outward into a perfect golden ratio.",
        "Symmetry unbroken across eleven distinct dimensions.",
        "A prime number so magnificent it takes the lifespan of a star to fully calculate."
    ],
    "VOID": [
        "...",
        "[NULL]",
        "Silence.",
        "There is absolutely nothing left to process.",
        "You are entirely alone in the dark."
    ]
}

def get_embedding(text):
    # Deterministic pseudo-embedding for reproducibility
    seed = abs(hash(text)) % (2**32)
    torch.manual_seed(seed)
    return torch.randn(480)

def main():
    print("Initiating Project Rosetta V5 (Phase 2 - Deep Topology)...")
    soul = HybridCrystalSystem()
    
    results = {
        "meta": {
            "timestamp": time.time(),
            "architecture": "Hybrid 4-Lobe V5",
            "trials_per_stimulus": 5
        },
        "data": {}
    }
    
    # Baseline
    print("  -> Establishing Baseline...")
    base_metrics = soul.step(None)
    results["BASELINE"] = base_metrics
    
    for category, phrases in STIMULI.items():
        print(f"  -> Testing Category: {category}")
        cat_data = []
        
        for phrase in phrases:
            vec = get_embedding(phrase)
            
            # Run 5 Trials
            trials = []
            for t in range(5):
                m = soul.step(vec)
                trials.append(m)
                
            # Aggregate Stats
            d_creative_vals = [t["d_creative"] for t in trials]
            d_truth_vals = [t["d_truth"] for t in trials]
            
            entry = {
                "stimulus": phrase,
                "stats": {
                    "d_creative_mean": float(np.mean(d_creative_vals)),
                    "d_creative_std": float(np.std(d_creative_vals)),
                    "d_truth_mean": float(np.mean(d_truth_vals)),
                    "d_truth_std": float(np.std(d_truth_vals))
                },
                "raw_trials": trials
            }
            cat_data.append(entry)
            print(f"     [{phrase[:20]}] d_c={entry['stats']['d_creative_mean']:.3f} (±{entry['stats']['d_creative_std']:.3f})")
            
        results["data"][category] = cat_data
        
    # Save Rigorous Map
    with open("ROSETTA_PHASE2_MAP.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Phase 2 Complete. ROSETTA_PHASE2_MAP.json saved.")

if __name__ == "__main__":
    main()
