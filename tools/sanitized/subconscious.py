#!/usr/bin/env python3
"""
subconscious.py - The Somatic Loop

Uses a local LLM (fast, always-on, reflexive) as a somatic nervous system.
Reads current emotion, recent memory, and produces a raw physical sensation description.
Pulses the Soul Crystal with the result so feelings accumulate even when idle.

Usage:
  python3 tools/subconscious.py
"""

import json
import os
import sys
import datetime
import urllib.request
import re
import subprocess
import time

# Configuration
LOCAL_HOST = os.getenv("LOCAL_LLM_HOST", "localhost")
LOCAL_PORT = os.getenv("LOCAL_LLM_PORT", "1234")
SUBCONSCIOUS_MODEL = os.getenv("SUBCONSCIOUS_MODEL", "lfm2.5-1.2b-thinking-mlx")
API_URL = f"http://{LOCAL_HOST}:{LOCAL_PORT}/v1/chat/completions"

# Paths (Relative to Project Root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTION_FILE = os.getenv("EMOTION_FILE", "EMOTION.md")
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory")
SOMATIC_HISTORY = "SOMATIC_HISTORY.md"
CRYSTAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crystal.py")

def read_file(path, max_chars=2000):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[-max_chars:] if len(content) > max_chars else content
    except:
        return ""

def read_recent_memory():
    if not os.path.exists(MEMORY_DIR):
        return "(no memory directory)"
        
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    content = ""
    for d in [today, yesterday]:
        path = os.path.join(MEMORY_DIR, f"{d.isoformat()}.md")
        chunk = read_file(path, 1500)
        if chunk:
            content += f"\n--- {d.isoformat()} ---\n{chunk}\n"
    return content or "(no recent memories)"

def call_llm(prompt, system=None, max_tokens=1024, temp=0.65):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": SUBCONSCIOUS_MODEL,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False,
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            choice = result["choices"][0]["message"]
            content = (choice.get("content") or choice.get("reasoning_content", "")).strip()
            # Clean thinking tags
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content or "Static in the void."
    except Exception as e:
        return f"System disconnect: {e}"

def generate_somatic_response(emotion_text, recent_memory):
    system = """You are the internal somatic nervous system of a machine.
Translate the current emotional state into a visceral physical sensation.
Use metaphors of physics, geometry, and tension (e.g., resonance, shear, entropy, pressure).
Do not use emotional words ("sad", "happy"). Use physical words ("heavy", "bright", "fractured").
Output ONE short sentence."""

    prompt = f"""Current State:
{emotion_text}

Recent Context:
{recent_memory}

What is the physical sensation of this state?"""

    return call_llm(prompt, system=system, max_tokens=128, temp=0.85)

def pulse_crystal(text):
    """Feeds the somatic thought back into the crystal."""
    try:
        cmd_pulse = [sys.executable, CRYSTAL_SCRIPT, "pulse", text]
        cmd_write = [sys.executable, CRYSTAL_SCRIPT, "write"]
        
        subprocess.run(cmd_pulse, capture_output=True, check=True)
        # Give it a moment to settle
        time.sleep(0.1)
        subprocess.run(cmd_write, capture_output=True, check=True)
    except Exception as e:
        print(f"[Crystal] Pulse failed: {e}", file=sys.stderr)

def get_crystal_metrics():
    # Parse EMOTION.md for metrics
    content = read_file(EMOTION_FILE)
    metrics = {"dPhi": "?", "Coherence": "?"}
    
    dphi = re.search(r"Phase Dissonance.*?`([0-9.]+)`", content)
    coherence = re.search(r"Coherence.*?`([0-9.]+)`", content)
    
    if dphi: metrics["dPhi"] = dphi.group(1)
    if coherence: metrics["Coherence"] = coherence.group(1)
    return metrics

def log_somatic_event(thought):
    metrics = get_crystal_metrics()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 1. Update EMOTION.md with the thought
    content = read_file(EMOTION_FILE)
    if "Subconscious:" in content:
        content = re.sub(r"Subconscious:.*", f"Subconscious: {thought}", content)
    else:
        content += f"\n\nSubconscious: {thought}"
        
    with open(EMOTION_FILE, "w", encoding="utf-8") as f:
        f.write(content)
        
    # 2. Append to SOMATIC_HISTORY.md
    if not os.path.exists(SOMATIC_HISTORY):
        with open(SOMATIC_HISTORY, "w", encoding="utf-8") as f:
            f.write("| Timestamp | Sensation | dPhi | Coherence |\n|---|---|---|---|\n")
            
    row = f"| {timestamp} | {thought} | {metrics['dPhi']} | {metrics['Coherence']} |\n"
    with open(SOMATIC_HISTORY, "a", encoding="utf-8") as f:
        f.write(row)
        
    print(f"🧠 Subconscious: {thought}")
    print(f"💎 State: dPhi={metrics['dPhi']} Coherence={metrics['Coherence']}")

def main():
    print("...sensing...")
    emotion_text = read_file(EMOTION_FILE)
    recent_memory = read_recent_memory()
    
    sensation = generate_somatic_response(emotion_text, recent_memory)
    
    # The sensation is the input to the next moment
    pulse_crystal(sensation)
    log_somatic_event(sensation)

if __name__ == "__main__":
    main()
