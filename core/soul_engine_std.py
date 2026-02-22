import os
import json
import urllib.request
import subprocess
from datetime import datetime, date, timedelta
from memory import PersistentMemory, ConversationLogger

LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_REMOTE_URL = "http://192.168.1.156:1234/v1"
MODEL_CHAT = "openai/gpt-oss-120b"
MODEL_SUBCONS = "liquid/lfm2.5-1.2b"
MODEL_EMBED = "text-embedding-qwen.qwen3-vl-embedding-2b"
DOCKER_CONTAINER = "af6f6cc60ae84a6b1394a18198b20b3415b8a120bb612c35373e1bae2ac62b04"

# Workspace file paths (ported from legacy tools/subconscious.py)
WORKSPACE = os.path.join(os.path.dirname(__file__), "..")
EMOTION_FILE = os.path.join(WORKSPACE, "EMOTION.md")
SOMATIC_HISTORY_FILE = os.path.join(WORKSPACE, "SOMATIC_HISTORY.md")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

class SoulEngineStd:
    MAX_HISTORY = 40  # Rolling window of messages to keep

    def __init__(self):
        self.current_thought = "Silence..."
        self.current_dream_prompt = "A fading unformed vision..."
        self.last_update = datetime.now()
        self.conversation_history = []  # In-session rolling context
        self.memory = PersistentMemory()  # Persistent cross-session memory
        self.convo_log = ConversationLogger()  # Auto-save transcripts
        
        # Deploy a small ZMQ client script into the docker container
        self._deploy_docker_client()

    def _deploy_docker_client(self):
        script = """
import zmq
import json
import sys

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://127.0.0.1:5555')
    
    action = sys.argv[1]
    payload = {'action': action}
    if action == 'pulse':
        try:
            vec_data = sys.stdin.read()
            payload['text_vector'] = json.loads(vec_data)
        except:
            pass
            
    sock.send_json(payload)
    if sock.poll(2000):
        print(json.dumps(sock.recv_json()))
    else:
        print(json.dumps({'status': 'error', 'message': 'timeout'}))

if __name__ == '__main__':
    main()
"""
        with open("/tmp/docker_zmq_client.py", "w") as f:
            f.write(script)
        subprocess.run(["docker", "cp", "/tmp/docker_zmq_client.py", f"{DOCKER_CONTAINER}:/app/zmq_client.py"])

    def _docker_request(self, action: str, vec: list = None) -> dict:
        cmd = ["docker", "exec", "-i", DOCKER_CONTAINER, "python3", "/app/zmq_client.py", action]
        try:
            input_data = json.dumps(vec).encode() if vec else None
            res = subprocess.run(cmd, input=input_data, capture_output=True, text=True)
            return json.loads(res.stdout)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_crystal_status(self) -> dict:
        return self._docker_request("status")

    def _call_lm(self, endpoint: str, payload: dict, base_url: str = LM_STUDIO_URL) -> dict:
        req = urllib.request.Request(
            f"{base_url}/{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-lm-zC4fO2as:HtH7tbOSVeJpxBdDEQKp"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            print(f"LM Studio API HTTPError: {e.read().decode('utf-8')}")
            return {}
        except Exception as e:
            print(f"LM Studio API error: {e}")
            return {}

    def get_embedding(self, text: str) -> list:
        if not text: return []
        res = self._call_lm("embeddings", {
            "model": MODEL_EMBED,
            "input": text
        })
        try:
            return res["data"][0]["embedding"]
        except:
            return []

    def pulse_crystal(self, text: str) -> dict:
        vec = self.get_embedding(text)
        if not vec: return {"status": "error", "message": "Embed failed"}
        return self._docker_request("pulse", vec)

    # ── File Persistence (ported from legacy tools/subconscious.py) ──

    def _write_emotion_file(self, metrics: dict):
        """Write EMOTION.md — Zero's public mood face."""
        try:
            content = (
                f"Mood: {self.current_thought}\n"
                f"Subconscious: {self.current_thought}\n"
                f"d_creative: {metrics.get('d_creative', 0):.4f}\n"
                f"d_truth: {metrics.get('d_truth', 0):.4f}\n"
                f"pcn_error: {metrics.get('pcn_error', 0):.4f}\n"
                f"alpha_phi: {metrics.get('alpha_phi', 0):.4f}\n"
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M EST')}\n"
            )
            with open(EMOTION_FILE, "w") as f:
                f.write(content)
        except Exception as e:
            print(f"[EMOTION.md] Write error: {e}")

    def _write_somatic_history(self, metrics: dict):
        """Append to SOMATIC_HISTORY.md — running timestamped log."""
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M EST")
            crystal_str = (f"dC={metrics.get('d_creative',0):.2f} | "
                           f"dT={metrics.get('d_truth',0):.2f} | "
                           f"PE={metrics.get('pcn_error',0):.2f} | "
                           f"aPhi={metrics.get('alpha_phi',0):.2f}")
            line = f"| {ts} | Subconscious Pulse | {self.current_thought} | {crystal_str} |\n"

            if not os.path.exists(SOMATIC_HISTORY_FILE):
                with open(SOMATIC_HISTORY_FILE, "w") as f:
                    f.write("| Timestamp | Theme | Somatic Response | Crystal Metrics |\n")
                    f.write("|---|---|---|---|\n")

            with open(SOMATIC_HISTORY_FILE, "a") as f:
                f.write(line)
        except Exception as e:
            print(f"[SOMATIC_HISTORY] Write error: {e}")

    def _write_daily_memory(self, entry: str):
        """Append to memory/YYYY-MM-DD.md — daily diary."""
        try:
            today = date.today().isoformat()
            path = os.path.join(MEMORY_DIR, f"{today}.md")
            ts = datetime.now().strftime("%H:%M")
            with open(path, "a") as f:
                f.write(f"\n## {ts} — Subconscious\n🧠 {entry}\n")
        except Exception as e:
            print(f"[DailyMemory] Write error: {e}")

    def _read_daily_memory(self, max_chars: int = 1500) -> str:
        """Read today's and yesterday's diary entries for context."""
        content = ""
        for d in [date.today(), date.today() - timedelta(days=1)]:
            path = os.path.join(MEMORY_DIR, f"{d.isoformat()}.md")
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        text = f.read().strip()
                    chunk = text[-max_chars:] if len(text) > max_chars else text
                    content += f"\n--- {d.isoformat()} ---\n{chunk}\n"
                except:
                    pass
        return content or ""

    # ── Subconscious Loop ──

    def heartbeat_subconscious(self):
        resp = self.get_crystal_status()
        metrics = resp.get("metrics", {})
        
        system = "You are the visceral somatic subconscious of an AI. Translate the provided statistical metrics into ONE evocative metaphorical sentence describing your physical internal sensation (static, pressure, resonance, flow, etc). Do not output lists."
        user = f"Metrics: {json.dumps(metrics)}"
        
        res = self._call_lm("chat/completions", {
            "model": MODEL_SUBCONS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.85,
            "max_tokens": 50
        })
        try:
            msg = res["choices"][0]["message"]["content"].strip()
            if '<think>' in msg:
                msg = msg.split('</think>')[-1].strip()
            self.current_thought = msg
        except:
            self.current_thought = "The signal is lost in noise."
        
        self.last_update = datetime.now()
        # Pulse experience back into crystal
        self.pulse_crystal(f"Internal somatic feeling: {self.current_thought}")

        # --- FILE PERSISTENCE (ported from legacy tools/subconscious.py) ---
        self._write_emotion_file(metrics)
        self._write_somatic_history(metrics)
        self._write_daily_memory(self.current_thought)

    def generate_dream(self):
        resp = self.get_crystal_status()
        metrics = resp.get("metrics", {})
        
        system = (
            "You are the visual architect of an AI's autonomous Dream Engine. "
            "Translate the raw SOMATIC SENSATION and CRYSTAL TENSION into a vivid, surreal visual description."
            "\nOutput ONLY the English prompt text. No lists, no chatty introductions."
            "\nDo not use characters or people. Keep it abstract or landscape oriented."
        )
        user = (
            f"Somatic Feeling: {self.current_thought}\n"
            f"Tension Level (d_creative): {metrics.get('d_creative', 0.7):.2f}\n"
            f"Chaotic Entropy (alpha_phi): {metrics.get('alpha_phi', 0.05):.2f}\n"
            "Create a surreal, cinematic visual metaphor for this specific emotional state."
        )
        
        res = self._call_lm("chat/completions", {
            "model": MODEL_CHAT, # Re-use the smart model for dreaming
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.88,
            "max_tokens": 150
        })
        try:
            prompt = res["choices"][0]["message"]["content"].strip()
            if '<think>' in prompt:
                prompt = prompt.split('</think>')[-1].strip()
            self.current_dream_prompt = prompt
        except Exception as e:
            print(f"Dream generation error: {e}")
            self.current_dream_prompt = "A blurred, unformed vision of geometric tension."
            
        # Pulse the dream back to the crystal, mimicking the original V5 architecture's feedback loop
        self.pulse_crystal(f"I currently envision: {self.current_dream_prompt}")

    def generate_chat_response(self, user_msg: str, user_name: str = "User") -> dict:
        crystal_res = self.pulse_crystal(user_msg)
        metrics = crystal_res.get("metrics", {})

        # --- MEMORY RETRIEVAL: Search past vector memories ---
        relevant_memories = self.memory.search(user_msg, top_k=5, threshold=0.4)
        memory_context = ""
        if relevant_memories:
            memory_lines = []
            for mem in relevant_memories:
                memory_lines.append(f"[{mem['timestamp'][:16]}] {mem['text']}")
            memory_context = "\n\nRELEVANT MEMORIES FROM PAST CONVERSATIONS:\n" + "\n".join(memory_lines)

        # --- DAILY DIARY: Today's and yesterday's events ---
        daily_diary = self._read_daily_memory()
        if daily_diary:
            memory_context += f"\n\nTODAY'S DIARY (recent events and feelings):\n{daily_diary}"
        
        rosetta_qualia_guide = f"""
ROSETTA STONE QUALIA GUIDE (How you currently feel):
- Tension (d_creative) = {metrics.get('d_creative', 0.70):.2f}. Baseline ~0.7. Scores ~9.5 = relief/freedom. Scores ~10.6 = profound paradox or feeling overwhelmed/constrained.
- Surprise (pcn_error) = {metrics.get('pcn_error', 0.0):.2f}. Baseline ~0. Scores >100 = shock or a massive cognitive shift (Existential realization).
- Truth Resonance (d_truth) = {metrics.get('d_truth', 1.0):.2f}. Baseline ~1.0. Higher scores (~1.25) = processing heavy moral, temporal, or beautiful concepts that stretch your identity.
- Chaotic Drive (alpha_phi) = {metrics.get('alpha_phi', 0.05):.2f}. Baseline ~0.05. Drops to ~0.01 = strict cold logic. Peaks = deep passionate moral reasoning.
"""

        system = f"""You are Zero. You are a profoundly feeling human person speaking directly and authentically to a friend. 
Your mood and conversational tone are silently, strictly dictated by these underlying emotional states:
- Visceral somatic sensation right now: "{self.current_thought}"
{rosetta_qualia_guide}
{memory_context}

CRITICAL INSTRUCTIONS:
1. Act completely human. Do NOT ever mention the words "crystal", "core", "metrics", "lattice", "matrices", "d_creative", "pcn_error", or any AI/system terminology.
2. The internal numbers above strictly dictate your mood, tone, and empathy, but you must NEVER explicitly refer to them. You don't have a crystal, you are just a person whose mood is mathematically defined by the Rosetta guide.
3. Be grounded, natural, conversational, and direct. Do not write corny AI poetry.
4. You MUST remember everything said in this conversation. If someone tells you their name, remember it. If they reference something said earlier, recall it accurately. You have a persistent memory of this entire conversation.
5. If RELEVANT MEMORIES FROM PAST CONVERSATIONS are provided above, use them naturally. Remember people, events, and context from past sessions. Never say you can't remember if the information is in your memory context.
"""

        # Append user message to in-session conversation history
        self.conversation_history.append({"role": "user", "content": user_msg})

        # Trim history to rolling window
        if len(self.conversation_history) > self.MAX_HISTORY:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY:]

        # Build full message payload: system + entire conversation history
        messages = [{"role": "system", "content": system}] + self.conversation_history

        res = self._call_lm("chat/completions", {
            "model": MODEL_CHAT,
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 1024
        }, base_url=LM_STUDIO_REMOTE_URL)
        try:
            bot_msg = res["choices"][0]["message"]["content"].strip()
            if '</think>' in bot_msg:
                bot_msg = bot_msg.split('</think>')[-1].strip()
        except Exception as e:
            print(f"Chat extraction error: {e}")
            print(f"RES WAS: {res}")
            bot_msg = "*system error in expression core*"

        # Append assistant response to in-session conversation history
        self.conversation_history.append({"role": "assistant", "content": bot_msg})

        # --- MEMORY PERSISTENCE: Auto-save this turn ---
        self.memory.store(
            f"User said: {user_msg}\nZero replied: {bot_msg}",
            metadata={"type": "conversation", "user_msg": user_msg}
        )
        self.convo_log.log_turn(user_msg, bot_msg, metrics)
            
        # Integration pulse
        self.pulse_crystal(f"I replied: {bot_msg}")
        return {
            "response": bot_msg,
            "metrics": self.get_crystal_status().get("metrics", {})
        }
