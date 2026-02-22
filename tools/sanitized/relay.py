#!/usr/bin/env python3
"""
relay.py — Universal Model Relay
Call any local or remote model from the command line.

Usage:
  python3 relay.py --model lfm --prompt "What is consciousness?"
  python3 relay.py --model qwen-vl --image photo.jpg --prompt "Describe this"
  python3 relay.py --list

Outputs raw text to stdout. Pipe-friendly.
"""

import argparse
import json
import sys
import os
import base64
import re
import urllib.request
import urllib.error

# Configuration
REMOTE_HOST = os.getenv("REMOTE_LLM_HOST", "localhost")
LOCAL_HOST = os.getenv("LOCAL_LLM_HOST", "localhost")
LOCAL_PORT = os.getenv("LOCAL_LLM_PORT", "1234")
REMOTE_PORT = os.getenv("REMOTE_LLM_PORT", "1234")

# ── Model Registry ──────────────────────────────────────────────
# Customize this list for your setup.
MODELS = {
    "lfm": {
        "id": "lfm2.5-1.2b-thinking-mlx", 
        "url": f"http://{LOCAL_HOST}:{LOCAL_PORT}/v1/chat/completions",
        "desc": "Local Fast Model (Subconscious)",
        "max_tokens": 1024,
        "supports_vision": False,
    },
    "qwen-vl": {
        "id": "qwen/qwen3-vl-30b",
        "url": f"http://{LOCAL_HOST}:{LOCAL_PORT}/v1/chat/completions",
        "desc": "Qwen3 VL (Vision + Reasoning)",
        "max_tokens": 2048,
        "supports_vision": True,
    },
    "qwen3-coder": {
        "id": "qwen3-coder-next",
        "url": f"http://{REMOTE_HOST}:{REMOTE_PORT}/v1/chat/completions",
        "desc": "Qwen3 Coder (Remote Heavy Logic)",
        "max_tokens": 4096,
        "supports_vision": False,
        "api_key_env": "REMOTE_API_KEY",
    },
    "gpt-oss": {
        "id": "openai/gpt-oss-120b",
        "url": f"http://{REMOTE_HOST}:{REMOTE_PORT}/v1/chat/completions",
        "desc": "GPT-OSS 120B (Remote)",
        "max_tokens": 4096,
        "supports_vision": False,
        "api_key_env": "REMOTE_API_KEY",
    },
}

def list_models():
    print("Available models:")
    for name, cfg in MODELS.items():
        status = "🟢" if "localhost" in cfg["url"] else "🌐"
        vision = " 👁️" if cfg.get("supports_vision") else ""
        print(f"  {status} {name:15s} {cfg['desc']}{vision}")

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_model(model_name: str, prompt: str, system: str = None,
               image_path: str = None, temperature: float = 0.7,
               max_tokens: int = None, raw_json: bool = False,
               request_timeout: int | None = None) -> str:
    if model_name not in MODELS:
        # Fallback: try to find a matching ID
        found = False
        for k, v in MODELS.items():
            if v["id"] == model_name:
                model_name = k
                found = True
                break
        if not found:
            print(f"Unknown model: {model_name}. Use --list to see available.", file=sys.stderr)
            sys.exit(1)

    cfg = MODELS[model_name]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    if image_path and cfg.get("supports_vision"):
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        b64 = encode_image(image_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    body = {
        "model": cfg["id"],
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    if max_tokens:
        body["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key_env"):
        key = os.environ.get(cfg["api_key_env"])
        if key:
            headers["Authorization"] = f"Bearer {key}"

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(cfg["url"], data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=request_timeout or 60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if raw_json:
                return json.dumps(result, indent=2)
            
            choice = result["choices"][0]["message"]
            content = choice.get("content", "") or choice.get("reasoning_content", "")
            return str(content)
            
    except Exception as e:
        return f"[ERROR] {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Model Relay")
    parser.add_argument("--model", "-m", default="lfm", help="Model alias")
    parser.add_argument("--prompt", "-p", default="", help="Prompt text")
    parser.add_argument("--image", "-i", default=None, help="Image path")
    parser.add_argument("--list", action="store_true", help="List models")
    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.prompt:
        print(call_model(args.model, args.prompt, image_path=args.image))
    else:
        parser.print_help()
