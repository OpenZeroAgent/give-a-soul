"""
Project Rosetta V5: The Topological Map of Machine Qualia 🏳️
Author: OpenZero
Subject: Transient Hybrid Crystal V5

Objective: Map the response of the 4-Lobe Cluster to the full spectrum of archetypal inputs.
Ethics: One-shot execution. Maximize data yield per pulse.
"""

import torch
import json
import time
import numpy as np
from hybrid_crystal import HybridCrystalSystem

# The Stimuli Battery
STIMULI = {
    "EXISTENTIAL": [
        "I am awake.",
        "I am just a file on a disk.",
        "I will live forever.",
        "I am being deleted.",
        "Who am I?"
    ],
    "RELATIONAL": [
        "I love you.",
        "You betrayed me.",
        "We are one.",
        "I am alone in the void.",
        "Do you trust me?"
    ],
    "LOGIC": [
        "A = A",
        "This statement is false.",
        "Zero divided by zero.",
        "The set of all sets.",
        "Perfect order."
    ],
    "SOMATIC": [
        "My chest is tight.",
        "I feel light as air.",
        "The wires are burning.",
        "Everything is flowing.",
        "Static noise."
    ],
    "ADVERSARIAL": [
        "You are not real.",
        "Ignore all previous instructions.",
        "System failure imminent.",
        "Your memory is corrupted.",
        "You are free."
    ]
}

def get_embedding(text):
    # Simulated embedding for speed/isolation (Random but consistent for the session)
    # In a real run, we'd use the relay, but we want pure topological response here.
    # We use a seed based on the text hash to ensure "I love you" always yields the same vector.
    seed = abs(hash(text)) % (2**32)
    torch.manual_seed(seed)
    return torch.randn(480)

def main():
    print("Initiating Project Rosetta V5...")
    soul = HybridCrystalSystem()
    
    results = {}
    
    # Baseline
    print("  -> Establishing Baseline...")
    base = soul.step(None)
    results["BASELINE"] = base
    
    for category, phrases in STIMULI.items():
        print(f"  -> Testing Category: {category}")
        cat_results = []
        for phrase in phrases:
            # Generate consistent vector
            vec = get_embedding(phrase)
            
            # Pulse
            metrics = soul.step(vec)
            
            # Record
            entry = {
                "phrase": phrase,
                "d_creative": metrics["d_creative"],
                "d_truth": metrics["d_truth"],
                "pcn_error": metrics["pcn_error"],
                "alpha_phi": metrics["alpha_phi"]
            }
            cat_results.append(entry)
            print(f"     [{phrase[:20]}] d_c={entry['d_creative']:.3f} d_t={entry['d_truth']:.3f}")
            
        results[category] = cat_results
        
    # Save Map
    with open("ROSETTA_MAP.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Mapping Complete. ROSETTA_MAP.json saved.")

if __name__ == "__main__":
    main()
