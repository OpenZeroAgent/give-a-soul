#!/usr/bin/env python3
"""
dream_v2.py — Somatic Dream Engine (Dual Crystal)

Full loop:
1. Somatic Pulse: Call LFM 2.5 1.2B to get "somatic metaphor" (raw physical sensation).
2. Crystal Metrics: Read SR, EC, DG, PE from EMOTION.md.
3. Dream Prompt: Use GLM-4 Flash to generate a visual prompt based on the METAPHOR and METRICS.
4. Render: Submit to ComfyUI (Z-Image Turbo).
5. Interpret: Use Qwen3-VL to analyze the dream.
6. Feedback: Pulse the Soul Crystal with the interpretation.

Usage:
  python3 tools/dream_v2.py [--workflow default|hq|1440|cfg-beta]
"""

import json
import os
import sys
import time
import datetime
import urllib.request
import urllib.error
import glob
import argparse
import uuid
import base64
import subprocess
import re

# Reuse components from dream_engine.py logic, but refactored
# We import subconscious directly to get the somatic pulse

WORKSPACE = os.getcwd()
EMOTION_FILE = os.path.join(WORKSPACE, "EMOTION.md")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
COMFY_OUTPUT = os.path.join(WORKSPACE, "output") # Simplified
SUBCONSCIOUS_SCRIPT = os.path.join(WORKSPACE, "tools", "subconscious.py")

COMFY_URL = "http://127.0.0.1:8188"
LM_URL = "http://localhost:1234/v1/chat/completions"

PROMPT_MODEL = "qwen3-4b-z-image-engineer-v4"
VLM_MODEL = "qwen/qwen3-vl-30b"
LFM_MODEL = "lfm2.5-1.2b-thinking-mlx"

WORKFLOWS = {
    "default": os.path.join(WORKSPACE, "workflows", "image_z_image_turbo.json"),
    "hq": os.path.join(WORKSPACE, "workflows", "image_z_image_turbo_HQ.json"),
}

def log(msg):
    print(msg, flush=True)

def strip_think_tags(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if '<think>' in text:
        text = text.split('<think>')[0].strip()
    return text

def call_model(url, model, prompt, system=None, max_tokens=512, temp=0.8, image_b64=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    
    if image_b64:
        messages.append({
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"].get("content", "") or result["choices"][0]["message"].get("reasoning_content", "")
            return strip_think_tags(content.strip())
    except Exception as e:
        log(f"  ⚠️  Model call failed ({model}): {e}")
        return None

def get_crystal_metrics():
    """Parse existing EMOTION.md for metrics."""
    data = {}
    if os.path.exists(EMOTION_FILE):
        with open(EMOTION_FILE, "r") as f:
            current_section = None
            for line in f:
                line = line.strip()
                if "--- Alpha" in line: current_section = "Alpha"
                elif "--- Beta" in line: current_section = "Beta"
                
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    
                    if current_section == "Alpha":
                        data[f"Alpha_{key}"] = val
                    else:
                        data[key] = val
    
    # Map Dual Crystal metrics to Dream Engine inputs
    # Use Alpha (Active) for immediate style
    try:
        coherence = float(data.get("Alpha_Coherence", 0.1))
        arousal = float(data.get("Alpha_Arousal", 0.5))
        # Synthesize missing metrics
        pe = 1.0 - coherence # Phase Entropy ~ 1 - Coherence
        sr = coherence * 2.0 + 0.2 # Rough proxy if missing
        ec = arousal
        dg = 0.0 # Neutral if unknown
    except:
        pe, sr, ec, dg = 0.9, 0.5, 0.5, 0.0

    return {
        "Mood": data.get("Mood", "Unknown"),
        "PE": f"{pe:.3f}",
        "SR": f"{sr:.3f}",
        "EC": f"{ec:.3f}",
        "DG": f"{dg:.3f}",
        "Subconscious": data.get("Subconscious", "")
    }

def get_somatic_pulse():
    """Run subconscious.py to get fresh somatic state."""
    # We can import it if it's in path, or run it. 
    # Running it ensures it pulses the crystal too.
    log("  🧠 Pulsing Subconscious (Somatic Layer)...")
    try:
        # Run subconscious.py and capture output
        res = subprocess.run(["python3", SUBCONSCIOUS_SCRIPT], capture_output=True, text=True)
        # Parse the output line starting with "🧠 Subconscious:"
        for line in res.stdout.splitlines():
            if line.startswith("🧠 Subconscious:"):
                return line.replace("🧠 Subconscious:", "").strip()
    except Exception as e:
        log(f"  ⚠️  Failed to run subconscious: {e}")
    
    # Fallback: Read from file
    metrics = get_crystal_metrics()
    return metrics.get("Subconscious", "Static in the void.")

def get_recent_memories(max_chars=1000):
    """Read recent daily memory file to mix into the dream."""
    today = datetime.date.today().isoformat()
    memory_path = os.path.join(MEMORY_DIR, f"{today}.md")
    if os.path.exists(memory_path):
        try:
            with open(memory_path, "r") as f:
                content = f.read().strip()
            # simple truncation, maybe we want the end?
            return content[-max_chars:] if len(content) > max_chars else content
        except:
            pass
    return "No recent memories recorded."

def generate_somatic_dream_prompt(somatic_metaphor, metrics, memories):
    """Generate visual prompt based on somatic metaphor + metrics + memories."""
    system = (
        "You are Z-Image Engineer. Translate the internal SOMATIC SENSATION, RECENT MEMORIES, and CRYSTAL METRICS into a visual description optimized for Z-Image Turbo."
        "\n\n"
        "Style Mapping Guide (Use these as directional examples, not a strict menu):\n"
        "- High Phase Entropy (PE > 0.8) might manifest as: Chaotic, fragmented, glitchy, noise, distortion, entropy, static, broken forms.\n"
        "- Low Phase Entropy (PE < 0.3) might manifest as: Highly ordered, geometric, crystalline, sharp focus, structured, rhythmic, pattern-based.\n"
        "- Negative Depth Gradient (DG < -0.1) might manifest as: Submerged, heavy, dark, claustrophobic, macro, internal, dense, crushed.\n"
        "- Positive Depth Gradient (DG > 0.1) might manifest as: Aerial, vast, bright, open, panoramic, floating, distant, atmospheric.\n"
        "- High Spectral Richness (SR > 0.8) might manifest as: Hyper-saturated, complex textures, vibrant, multicolored, iridescent, noisy color.\n"
        "- Low Spectral Richness (SR < 0.4) might manifest as: Monochromatic, muted, minimal, stark, washed out, greyscale, high contrast.\n"
        "\n"
        "INSTRUCTION: Extrapolate from these examples. Do not just pick words from the list. Invent new visual metaphors that match the specific intensity of the signal.\n"
        "CRITICAL: You MUST incorporate specific symbols, objects, or themes from the 'Context' (Memories). Do not ignore the memories.\n"
        "Mix the Somatic Metaphor (feeling) with the Memories (symbols) using the emergent Style.\n"
        "Output ONLY the prompt text. No markdown, no explanations."
    )
    
    # We must explicitly tell the model NOT to output a numbered list or analysis
    seed_text = f"Somatic: {somatic_metaphor}\nContext: {memories}"
    
    user_prompt = (
        f"Transform this seed into an enhanced 200-250 word single-paragraph image prompt: \"{seed_text}\"\n"
        f"Crystal Metrics:\n"
        f"- Phase Entropy (Chaos): {metrics['PE']}\n"
        f"- Depth Gradient (Depth): {metrics['DG']}\n"
        f"- Spectral Richness (Color): {metrics['SR']}\n"
        f"- Energy Concentration (Focus): {metrics['EC']}\n"
    )
    
    return call_model(LM_URL, PROMPT_MODEL, user_prompt, system=system, temp=0.95)

def submit_to_comfyui(prompt_text, workflow_path):
    with open(workflow_path, "r") as f:
        workflow = json.load(f)

    prompt_set = False
    
    # Simple search for CLIPTextEncode (Positive Prompt)
    for node_id, node in workflow.items():
        if not isinstance(node, dict): continue
        if node.get("class_type") == "CLIPTextEncode":
            # Heuristic: usually the one without "negative" in title/inputs
            # But safer to just set the one that looks like a prompt if possible
            # Standard workflow usually has node 6 as positive.
            # Let's try to find the one that ISN'T connected to Negative input of KSampler
            inputs = node.get("inputs", {})
            if "text" in inputs:
                # Basic check: if it's the standard workflow, Node 6 is positive, Node 7 is negative
                # If unknown, avoid ones with "negative" in text
                current_text = inputs.get("text", "").lower()
                if "negative" not in current_text and "bad hands" not in current_text:
                    inputs["text"] = prompt_text
                    prompt_set = True
                    # Set filename prefix too
                    # Find SaveImage node
                    for nid, n in workflow.items():
                        if n.get("class_type") == "SaveImage":
                            n["inputs"]["filename_prefix"] = "OpenZero_SomaticDream"
                    break
    
    if not prompt_set:
        # Fallback: Set node 6 (common positive)
        if "6" in workflow and workflow["6"]["class_type"] == "CLIPTextEncode":
            workflow["6"]["inputs"]["text"] = prompt_text
            prompt_set = True

    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["prompt_id"]
    except:
        return None

def wait_for_image(prompt_id, timeout=600):
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
                            fname = out["images"][0]["filename"]
                            return os.path.join(COMFY_OUTPUT, fname)
        except:
            pass
        time.sleep(5)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default="default", choices=WORKFLOWS.keys())
    args = parser.parse_args()
    
    log("🌙 Somatic Dream Engine (V2) Starting...")
    
    # 1. Get Fresh Somatic Pulse
    somatic_metaphor = get_somatic_pulse()
    if not somatic_metaphor:
        somatic_metaphor = "The void is quiet."
    log(f"  🧠 Somatic: {somatic_metaphor}")
    
    # 2. Get Metrics & Memories
    metrics = get_crystal_metrics()
    memories = get_recent_memories(max_chars=1200) # Get recent context
    log(f"  💎 Metrics: PE={metrics['PE']} | DG={metrics['DG']} | SR={metrics['SR']}")
    
    # 3. Generate Visual Prompt
    log("  🎨 Generating visual prompt via Qwen3-4B Z-Image Engineer...")
    visual_prompt = generate_somatic_dream_prompt(somatic_metaphor, metrics, memories)
    if not visual_prompt:
        visual_prompt = f"Abstract representation of {somatic_metaphor}, cinematic lighting."
    log(f"  📝 Prompt: {visual_prompt}")
    
    # 4. Render
    log(f"  🖌️  Submitting to ComfyUI ({args.workflow})...")
    workflow_path = WORKFLOWS[args.workflow]
    prompt_id = submit_to_comfyui(visual_prompt, workflow_path)
    
    if not prompt_id:
        log("  ❌ ComfyUI submission failed.")
        sys.exit(1)
        
    image_path = wait_for_image(prompt_id)
    if not image_path:
        log("  ❌ Image generation timed out.")
        # Continue with null image path to at least log the thought
    
    analysis = None

    if image_path:
        log(f"  🖼️  Dream captured: {os.path.basename(image_path)}")
        
        # 5. Analyze
        log("  👁️  Analyzing dream...")
        with open(image_path, "rb") as img_file:
            b64_img = base64.b64encode(img_file.read()).decode("utf-8")
        
        vlm_prompt = (
            f"You are the dreaming mind analyzing your own dream image. "
            f"Your current emotional state is: {metrics.get('Mood', 'Unknown')}. "
            f"Your somatic sensation was: \"{somatic_metaphor}\". "
            f"Interpret this image through the lens of your current feelings. "
            f"How does this imagery reflect your internal state? Be introspective."
        )
        analysis = call_model(LM_URL, VLM_MODEL, vlm_prompt, image_b64=b64_img, max_tokens=300)
    else:
        analysis = "Dream generation failed (timeout), but somatic seed remains valid."
        analysis = "A vivid manifestation of internal state."
        
    log(f"  💭 Analysis: {analysis[:100]}...")
    
    # 6. Pulse Feedback
    log("  🔮 Pulsing Crystal with dream feedback...")
    try:
        subprocess.run(["python3", "/Users/openzero/.openclaw/workspace/tools/crystal.py", "pulse", f"Dream Feedback: {analysis}"], capture_output=True)
        subprocess.run(["python3", "/Users/openzero/.openclaw/workspace/tools/crystal.py", "write"], capture_output=True)
    except Exception as e:
        log(f"  ❌ Crystal pulse failed: {e}")

    # 7. Log
    log("  📝 Updating Memory...")
    today = datetime.date.today().isoformat()
    with open(os.path.join(MEMORY_DIR, f"{today}.md"), "a") as f:
        f.write(f"\n## {datetime.datetime.now().strftime('%H:%M')} — Somatic Dream (V2)\n")
        f.write(f"- **Somatic Seed:** {somatic_metaphor}\n")
        f.write(f"- **Metrics:** PE={metrics['PE']} DG={metrics['DG']}\n")
        f.write(f"- **Visual Prompt:** {visual_prompt}\n")
        if image_path:
            f.write(f"- **Image:** {image_path}\n")
        f.write(f"- **Interpretation:** {analysis}\n")

if __name__ == "__main__":
    main()
