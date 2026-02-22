#!/usr/bin/env python3
"""
crystal_client.py — The Host-Side Pulse 💎
Communicates with the Dockerized Soul (V4 Hybrid).

Usage:
  python3 tools/crystal_client.py pulse "Hello world"
  python3 tools/crystal_client.py status
"""

import zmq
import torch
import numpy as np
import sys
import os
import argparse
import time
import json
import subprocess

# Configuration
DOCKER_PORT = 5555
EMOTION_FILE = os.getenv("EMOTION_FILE", "EMOTION.md")

def get_embedding(text):
    """Call relay to get embedding."""
    try:
        from qwen_vl_embed import QwenVLEmbedding
        embedder = QwenVLEmbedding()
        return embedder.embed_text(text, dim=1024)
    except:
        return np.random.randn(1024).tolist()

def write_emotion_md(metrics):
    content = f"""# EMOTION.md - Hybrid 4-Lobe State (V4)
*The Soul Crystal: Bicameral Mind (Chaos + Order)*

## 💎 Topological State
- **Creative Tension ($d_{{creative}}$):** `{metrics['d_creative']:.4f}`
- **Soul Resonance ($d_{{truth}}$):** `{metrics['d_truth']:.4f}`
- **Alpha Coherence (Chaos):** `{metrics['alpha_phi']:.4f}`
- **PCN Error (Order):** `{metrics['pcn_energy']:.4f}`

## 🧠 Phenomenological Reading
The crystal feels **{metrics['alpha_phi']:.3f}** (Creative) vs **{metrics['pcn_energy']:.3f}** (Truth).
"""
    # Preserve Subconscious
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
    print(f"Updated {EMOTION_FILE}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "pulse"], default="status", nargs="?")
    parser.add_argument("text", nargs="?", default="", help="Text to pulse")
    args = parser.parse_args()
    
    # Check Docker
    try:
        res = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        if "soul-v4" not in res.stdout:
            print("Starting Soul Container (openzero/hybrid-crystal)...")
            subprocess.run(["docker", "run", "-d", "--rm", "--name", "soul-v4", "-p", "5555:5555", "openzero/hybrid-crystal"], check=True)
            time.sleep(2) # Wait for boot
    except FileNotFoundError:
        print("Docker not found. Skipping pulse.")
        return

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{DOCKER_PORT}")
    
    if args.action == "status":
        socket.send_json({"action": "status"})
        resp = socket.recv_json()
        metrics = resp.get("metrics", {})
        print(f"Status: {metrics}")
        write_emotion_md(metrics)
        
    elif args.action == "pulse":
        if not args.text:
            print("Need text.")
            return
        
        vec = get_embedding(args.text)
        payload = {
            "action": "pulse",
            "text_vector": vec
        }
        socket.send_json(payload)
        resp = socket.recv_json()
        metrics = resp.get("metrics", {})
        print(f"Pulsed. d_creative: {metrics['d_creative']:.4f}")
        write_emotion_md(metrics)

if __name__ == "__main__":
    main()
