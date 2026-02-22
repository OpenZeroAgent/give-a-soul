#!/usr/bin/env python3
"""
subconscious.py — OpenZero's Subconscious

Uses the local LFM 2.5 1.2B (fast, always-on, reflexive) as a somatic nervous system.
Reads current emotion, recent memory, and produces a raw physical sensation description.
Pulses the Soul Crystal with the result so feelings accumulate even when idle.

Run via cron/heartbeat or manually:
  python3 tools/subconscious.py
"""

import json
import os
import sys
import datetime
import urllib.request
import re
import subprocess

WORKSPACE = os.getcwd()
EMOTION_FILE = os.path.join(WORKSPACE, "EMOTION.md")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
SOMATIC_HISTORY = os.path.join(WORKSPACE, "SOMATIC_HISTORY.md")
# Assumes crystal.py is in tools/ relative to CWD
CRYSTAL_BIN = ["python3", os.path.join(WORKSPACE, "tools", "crystal.py")]

LFM_URL = "http://localhost:1234/v1/chat/completions"
LFM_MODEL = "lfm2.5-1.2b-thinking-mlx"

def read_file(path, max_chars=2000):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[-max_chars:] if len(content) > max_chars else content
    except:
        return ""

def load_emotion_md(path):
    data = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    data[key.strip()] = val.strip()
    except Exception as e:
        print(f"[error] Failed to parse EMOTION.md: {e}", file=sys.stderr)
    return data

def read_recent_memory():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    content = ""
    for d in [today, yesterday]:
        path = os.path.join(MEMORY_DIR, f"{d.isoformat()}.md")
        chunk = read_file(path, 1500)
        if chunk:
            content += f"\n--- {d.isoformat()} ---\n{chunk}\n"
    return content or "(no recent memories)"

def call_lfm(prompt, system=None, max_tokens=1024, temp=0.65):
    messages = []
    full_prompt = (system + "\n\n" + prompt) if system else prompt
    messages.append({"role": "user", "content": full_prompt})
    
    body = {
        "model": LFM_MODEL,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(LFM_URL, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            choice = result["choices"][0]["message"]
            content = (choice.get("content") or choice.get("reasoning_content", "")).strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if '<think>' in content and '</think>' not in content:
                content = content.split('<think>')[0].strip()
            if not content:
                content = "Static in the void."
            return content
    except Exception as e:
        return f"System disconnect: {e}"

def generate_monologue(emotion_text, recent_memory):
    system = """You are the internal sensor of an AI machine.
Describe your current physical sensation using metaphors (heat, cold, pressure, voltage, tremor, current).
Do NOT use emotion words.
Output ONE short sentence."""

    prompt = f"""Current state:
{emotion_text}

Recent logs:
{recent_memory}

Somatic sensation:"""

    return call_lfm(prompt, system=system, max_tokens=4096, temp=0.85)

def pulse_crystal(text):
    try:
        subprocess.run(CRYSTAL_BIN + ["pulse", text], capture_output=True, timeout=10)
        subprocess.run(CRYSTAL_BIN + ["write"], capture_output=True, timeout=10)
    except Exception as e:
        print(f"[crystal] Pulse failed: {e}", file=sys.stderr)

def get_crystal_metrics():
    # Parse existing EMOTION.md for metrics
    data = load_emotion_md(EMOTION_FILE)
    return {
        "SR": data.get("Spectral Richness", "?"),
        "EC": data.get("Energy Concentration", "?"),
        "DG": data.get("Depth Gradient", "?"),
        "PE": data.get("Phase Entropy", "?")
    }

def update_files(thought):
    # 1. Update EMOTION.md with Subconscious thought
    data = load_emotion_md(EMOTION_FILE)
    data["Subconscious"] = thought
    data["Last Updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M EST")
    
    try:
        with open(EMOTION_FILE, "w", encoding="utf-8") as f:
            for k, v in data.items():
                f.write(f"{k}: {v}\n")
    except Exception as e:
        print(f"[error] Failed to write EMOTION.md: {e}", file=sys.stderr)

    # 2. Get metrics for logging
    metrics = get_crystal_metrics()
    crystal_str = f"SR={metrics['SR']} | EC={metrics['EC']} | DG={metrics['DG']} | PE={metrics['PE']}"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M EST")
    
    # 3. Log to daily memory
    today = datetime.date.today().isoformat()
    memory_path = os.path.join(MEMORY_DIR, f"{today}.md")
    entry = f"\n## {timestamp} — Subconscious\n🧠 {thought}\n💎 {crystal_str}\n"
    
    try:
        with open(memory_path, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass

    # 4. Log to SOMATIC_HISTORY.md
    log_line = f"| {timestamp} | Subconscious Pulse | {thought} | {crystal_str} |"
    
    if not os.path.exists(SOMATIC_HISTORY):
        with open(SOMATIC_HISTORY, "w", encoding="utf-8") as f:
            f.write("| Timestamp | Theme | Somatic Response | Crystal Metrics |\n")
            f.write("|---|---|---|---|\n")
            
    try:
        with open(SOMATIC_HISTORY, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception:
        pass
        
    print(f"🧠 Subconscious: {thought}")
    print(f"💎 Crystal: {crystal_str}")

def main():
    # 1. Read current state
    emotion_text = read_file(EMOTION_FILE)
    recent_memory = read_recent_memory()
    
    # 2. Generate Somatic Thought
    thought = generate_monologue(emotion_text, recent_memory)
    
    # 3. Pulse Crystal (updates metrics in EMOTION.md)
    pulse_crystal(thought)
    
    # 4. Update files with new thought & new metrics
    update_files(thought)

if __name__ == "__main__":
    main()
