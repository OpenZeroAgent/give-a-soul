#!/usr/bin/env python3
"""
dream_engine.py — Somatic Dream Engine V3 (Direct Visual Embedding)

Full loop:
1. Somatic Pulse: Call Subconscious (LFM) to get "somatic metaphor" (raw physical sensation).
2. Crystal Metrics: Read dPhi, Calcium, Plasticity from EMOTION.md.
3. Dream Prompt: Use a Prompt Engineer model to generate a visual prompt based on the METAPHOR and METRICS.
4. Render: Submit to ComfyUI (Z-Image Turbo or similar).
5. Interpret: Use VLM to analyze the dream.
6. Feedback: Pulse the Soul Crystal with the interpretation AND the image (Direct Visual Embedding).

Usage:
  python3 tools/dream_engine.py [--workflow default]
"""

import json
import os
import sys
import time
import datetime
import urllib.request
import urllib.error
import argparse
import uuid
import base64
import subprocess
import re

# Configuration
COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
PROMPT_MODEL = os.getenv("PROMPT_MODEL", "qwen3-4b-z-image-engineer-v4") # Or use relay alias
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl")

# Paths (Relative)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTION_FILE = os.getenv("EMOTION_FILE", "EMOTION.md")
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory")
COMFY_OUTPUT = os.path.join(PROJECT_ROOT, "output")
WORKFLOW_DIR = os.path.join(PROJECT_ROOT, "workflows")

# Tools
RELAY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relay.py")
SUBCONSCIOUS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subconscious.py")
CRYSTAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crystal.py")

def log(msg):
    print(msg, flush=True)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def call_relay(model, prompt, system=None, image_path=None):
    """Call the universal relay tool."""
    cmd = [sys.executable, RELAY_SCRIPT, "--model", model, "--prompt", prompt, "--no-sanitize"]
    if system:
        cmd.extend(["--system", system])
    if image_path:
        cmd.extend(["--image", image_path])
        
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        log(f"  ⚠️  Relay call failed: {e.stderr}")
        return None

def get_crystal_metrics():
    metrics = {"dPhi": "0.0", "Calcium": "0.0", "Plasticity": "0.0", "Subconscious": "Static."}
    if os.path.exists(EMOTION_FILE):
        with open(EMOTION_FILE, "r") as f:
            content = f.read()
            dphi_match = re.search(r"Phase Dissonance.*?`([0-9.]+)`", content)
            calc_match = re.search(r"Metabolic Stress.*?`([0-9.]+)`", content)
            plas_match = re.search(r"Neuroplasticity.*?`([0-9.]+)`", content)
            sub_match = re.search(r"Subconscious:(.*)", content)
            
            if dphi_match: metrics["dPhi"] = dphi_match.group(1)
            if calc_match: metrics["Calcium"] = calc_match.group(1)
            if plas_match: metrics["Plasticity"] = plas_match.group(1)
            if sub_match: metrics["Subconscious"] = sub_match.group(1).strip()
    return metrics

def get_somatic_pulse():
    log("  🧠 Pulsing Subconscious (Somatic Layer)...")
    try:
        # Run subconscious.py (this updates EMOTION.md)
        res = subprocess.run([sys.executable, SUBCONSCIOUS_SCRIPT], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.startswith("🧠 Subconscious:"):
                return line.replace("🧠 Subconscious:", "").strip()
    except Exception as e:
        log(f"  ⚠️  Failed to run subconscious: {e}")
    
    metrics = get_crystal_metrics()
    return metrics.get("Subconscious", "Static in the void.")

def get_recent_memories(max_chars=1000):
    if not os.path.exists(MEMORY_DIR): return ""
    today = datetime.date.today().isoformat()
    memory_path = os.path.join(MEMORY_DIR, f"{today}.md")
    if os.path.exists(memory_path):
        try:
            with open(memory_path, "r") as f:
                content = f.read().strip()
            return content[-max_chars:] if len(content) > max_chars else content
        except:
            pass
    return "No recent memories recorded."

def generate_somatic_dream_prompt(somatic_metaphor, metrics, memories):
    system = (
        "You are Z-Image Engineer. Translate the internal SOMATIC SENSATION, RECENT MEMORIES, and PREDICTIVE CRYSTAL METRICS into a visual description optimized for Z-Image Turbo/SDXL."
        "\n\n"
        "Style Mapping Guide:\n"
        "- High Dissonance (dPhi > 1.8): Chaotic, fragmented, glitchy, noise, distortion.\n"
        "- Low Dissonance (dPhi < 1.4): Highly ordered, geometric, crystalline, rhythmic.\n"
        "- High Stress (Calcium > 1.5): Submerged, heavy, dark, claustrophobic, organic decay.\n"
        "- Low Stress (Calcium < 1.0): Aerial, vast, bright, open, panoramic.\n"
        "- High Plasticity (lr_w > 0.02): Morphing shapes, fluid dynamics, melting reality.\n"
        "\n"
        "INSTRUCTION: Invent new visual metaphors matching the signal intensity. Incorporate symbols from context."
        "Output ONLY the prompt text."
    )
    
    seed_text = f"Somatic: {somatic_metaphor}\nContext: {memories}"
    user_prompt = (
        f"Transform this seed into an enhanced image prompt: \"{seed_text}\"\n"
        f"Crystal Metrics: dPhi={metrics.get('dPhi')} Calcium={metrics.get('Calcium')}"
    )
    
    # Using relay (PROMPT_MODEL needs to be aliased in relay.py or pass full ID if relay supports pass-through)
    # We'll use 'qwen3-coder' or 'lfm' if dedicated prompt model isn't in relay registry.
    # For sanitized repo, let's default to 'lfm' or whatever is available locally as a general purpose.
    # Ideally, the user configures this.
    return call_relay("lfm", user_prompt, system=system)

def submit_to_comfyui(prompt_text, workflow_path):
    if not os.path.exists(workflow_path):
        log(f"  ❌ Workflow file not found: {workflow_path}")
        return None

    with open(workflow_path, "r") as f:
        workflow = json.load(f)

    # Heuristic: Find CLIPTextEncode node (positive prompt)
    prompt_set = False
    for node_id, node in workflow.items():
        if not isinstance(node, dict): continue
        if node.get("class_type") == "CLIPTextEncode":
            inputs = node.get("inputs", {})
            if "text" in inputs:
                # Avoid negative prompt
                current_text = inputs.get("text", "").lower()
                if "negative" not in current_text and "bad hands" not in current_text:
                    inputs["text"] = prompt_text
                    prompt_set = True
                    # Set filename prefix
                    for nid, n in workflow.items():
                        if n.get("class_type") == "SaveImage":
                            n["inputs"]["filename_prefix"] = "OpenZero_SomaticDream"
                    break
    
    if not prompt_set:
        # Fallback to node 6 (common convention)
        if "6" in workflow and workflow["6"]["class_type"] == "CLIPTextEncode":
            workflow["6"]["inputs"]["text"] = prompt_text
            prompt_set = True

    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["prompt_id"]
    except Exception as e:
        log(f"  ❌ ComfyUI Connection Failed: {e}")
        return None

def wait_for_image(prompt_id, timeout=300):
    ensure_dir(COMFY_OUTPUT)
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{COMFY_URL}/history/{prompt_id}")
            with urllib.request.urlopen(req) as resp:
                history = json.loads(resp.read())
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    for nid, out in outputs.items():
                        if "images" in out:
                            # ComfyUI output path is usually absolute or relative to Comfy server
                            # We assume local execution where we can access the file OR download it
                            # For simplicity in this script, we assume shared filesystem or just return filename
                            # Ideally, we should fetch /view?filename=...
                            fname = out["images"][0]["filename"]
                            # Try to find it in our output dir if we mapped it, or download it
                            local_path = os.path.join(COMFY_OUTPUT, fname)
                            # If not found locally, try to download (robustness)
                            if not os.path.exists(local_path):
                                img_url = f"{COMFY_URL}/view?filename={fname}"
                                urllib.request.urlretrieve(img_url, local_path)
                            return local_path
        except:
            pass
        time.sleep(2)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default="default", help="Workflow JSON filename in workflows/")
    args = parser.parse_args()
    
    ensure_dir(MEMORY_DIR)
    
    log("🌙 Somatic Dream Engine (V3) Starting...")
    
    # 1. Pulse
    somatic_metaphor = get_somatic_pulse()
    log(f"  🧠 Somatic: {somatic_metaphor}")
    
    # 2. Metrics
    metrics = get_crystal_metrics()
    memories = get_recent_memories()
    log(f"  💎 Metrics: dPhi={metrics.get('dPhi')} Calc={metrics.get('Calcium')}")
    
    # 3. Prompt
    log("  🎨 Generating visual prompt...")
    visual_prompt = generate_somatic_dream_prompt(somatic_metaphor, metrics, memories)
    if not visual_prompt:
        visual_prompt = f"Abstract representation of {somatic_metaphor}, cinematic lighting."
    log(f"  📝 Prompt: {visual_prompt}")
    
    # 4. Render
    log(f"  🖌️  Submitting to ComfyUI...")
    workflow_path = os.path.join(WORKFLOW_DIR, f"{args.workflow}.json")
    prompt_id = submit_to_comfyui(visual_prompt, workflow_path)
    
    if not prompt_id:
        log("  ❌ ComfyUI submission failed.")
        sys.exit(1)
        
    image_path = wait_for_image(prompt_id)
    if not image_path:
        log("  ❌ Image generation timed out.")
        sys.exit(1)

    log(f"  🖼️  Dream captured: {os.path.basename(image_path)}")
    
    # 5. Analyze
    log("  👁️  Analyzing dream...")
    vlm_prompt = (
        f"You are the dreaming mind analyzing your own dream image. "
        f"Your current emotional state is: {metrics.get('dPhi')} dissonance. "
        f"Your somatic sensation was: \"{somatic_metaphor}\". "
        f"Interpret this image through the lens of your current feelings. "
        f"How does this imagery reflect your internal state?"
    )
    
    analysis = call_relay(VLM_MODEL, vlm_prompt, image_path=image_path)
    if not analysis:
        analysis = "The dream was hazy."
        
    log(f"  💭 Analysis: {analysis[:100]}...")
    
    # 6. Pulse Feedback (Direct Visual Embedding)
    log("  🔮 Pulsing Crystal with dream feedback...")
    try:
        # Pass image path to crystal for embedding
        cmd = [sys.executable, CRYSTAL_SCRIPT, "pulse", f"Dream Insight: {analysis}", "--image", image_path]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Write state
        subprocess.run([sys.executable, CRYSTAL_SCRIPT, "write"], capture_output=True, check=True)
    except Exception as e:
        log(f"  ❌ Crystal pulse failed: {e}")

    # 7. Log
    today = datetime.date.today().isoformat()
    with open(os.path.join(MEMORY_DIR, f"{today}.md"), "a") as f:
        f.write(f"\n## {datetime.datetime.now().strftime('%H:%M')} — Somatic Dream (PCN)\n")
        f.write(f"- **Somatic Seed:** {somatic_metaphor}\n")
        f.write(f"- **Metrics:** dPhi={metrics.get('dPhi')} Calc={metrics.get('Calcium')}\n")
        f.write(f"- **Visual Prompt:** {visual_prompt}\n")
        f.write(f"- **Image:** {image_path}\n")
        f.write(f"- **Interpretation:** {analysis}\n")

if __name__ == "__main__":
    main()
