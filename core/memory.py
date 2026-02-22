"""
memory.py — Persistent Searchable Memory for Zero

Inspired by Agent Zero's FAISS-based vector memory system.
Uses our existing LM Studio embedding API + numpy cosine similarity.
Stores memories as JSON on disk with their embeddings for fast retrieval.

Architecture:
  - Each memory entry = { text, embedding, timestamp, metadata }
  - On save: auto-persist to disk as JSON
  - On search: cosine similarity against all stored embeddings
  - On boot: auto-load from disk
"""

import os
import json
import math
import urllib.request
from datetime import datetime

LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_EMBED = "text-embedding-qwen.qwen3-vl-embedding-2b"
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "vector_memory.json")
CONVO_LOG_DIR = os.path.join(MEMORY_DIR, "conversations")

# Ensure directories exist
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(CONVO_LOG_DIR, exist_ok=True)


def _get_embedding(text: str) -> list:
    """Get text embedding from LM Studio."""
    if not text:
        return []
    payload = json.dumps({"model": MODEL_EMBED, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('LM_STUDIO_API_KEY', 'lm-studio')}"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["data"][0]["embedding"]
    except Exception as e:
        print(f"[Memory] Embedding error: {e}")
        return []


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors using pure math."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class PersistentMemory:
    """
    Vector-based persistent memory store.
    Stores memories with embeddings for semantic search.
    Auto-saves to disk after every write.
    """

    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.memories: list[dict] = []
        self._load()

    def _load(self):
        """Load memories from disk."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.memories = json.load(f)
                print(f"[Memory] Loaded {len(self.memories)} memories from disk.")
            except Exception as e:
                print(f"[Memory] Failed to load: {e}")
                self.memories = []
        else:
            self.memories = []

    def _save(self):
        """Persist memories to disk."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            print(f"[Memory] Save error: {e}")

    def store(self, text: str, metadata: dict = None) -> bool:
        """Store a memory with its embedding. Auto-saves to disk."""
        embedding = _get_embedding(text)
        if not embedding:
            return False

        entry = {
            "text": text,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.memories.append(entry)
        self._save()
        return True

    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
        """Search memories by semantic similarity. Returns top_k results above threshold."""
        if not self.memories:
            return []

        query_embedding = _get_embedding(query)
        if not query_embedding:
            return []

        # Score all memories
        scored = []
        for mem in self.memories:
            if not mem.get("embedding"):
                continue
            score = _cosine_similarity(query_embedding, mem["embedding"])
            if score >= threshold:
                scored.append({
                    "text": mem["text"],
                    "score": score,
                    "timestamp": mem["timestamp"],
                    "metadata": mem.get("metadata", {})
                })

        # Sort by similarity score (highest first)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get the N most recent memories."""
        return [
            {"text": m["text"], "timestamp": m["timestamp"], "metadata": m.get("metadata", {})}
            for m in self.memories[-n:]
        ]

    def count(self) -> int:
        return len(self.memories)

    def clear(self):
        """Clear all memories. Use with caution."""
        self.memories = []
        self._save()


class ConversationLogger:
    """
    Auto-saves full conversation transcripts to disk.
    Each session gets its own timestamped file.
    """

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(CONVO_LOG_DIR, f"session_{self.session_id}.json")
        self.turns: list[dict] = []

    def log_turn(self, user_msg: str, assistant_msg: str, metrics: dict = None):
        """Log a single conversation turn."""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "assistant": assistant_msg,
            "metrics": metrics or {}
        }
        self.turns.append(turn)
        self._save()

    def _save(self):
        try:
            with open(self.log_file, "w") as f:
                json.dump({
                    "session_id": self.session_id,
                    "started": self.turns[0]["timestamp"] if self.turns else "",
                    "turns": self.turns
                }, f, indent=2)
        except Exception as e:
            print(f"[ConvoLog] Save error: {e}")

    def get_transcript(self) -> str:
        """Get full conversation as text."""
        lines = []
        for t in self.turns:
            lines.append(f"User: {t['user']}")
            lines.append(f"Zero: {t['assistant']}")
        return "\n".join(lines)
