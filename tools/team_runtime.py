#!/usr/bin/env python3
"""
Team Runtime Protocol - Helper Module
Implements the TEAM protocol messaging bus and schema validation for OpenClaw.
"""

import json
import os
import uuid
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Constants
WORKSPACE_DIR = os.path.expanduser("~/.openclaw/workspace")
TEAM_RUNTIME_DIR = os.path.join(WORKSPACE_DIR, "team_runtime")
SESSIONS_DIR = os.path.join(TEAM_RUNTIME_DIR, "sessions")

# Ensure directories exist
os.makedirs(SESSIONS_DIR, exist_ok=True)

@dataclass
class TeamMessage:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    sender: str = "Zero"  # Maps to "from" in JSON
    recipient: str = "ALL"  # Maps to "to" in JSON
    type: str = "text"  # text, proposal, critique, lock, system
    confidence: float = 1.0
    payload: Any = ""

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "from": self.sender,
            "to": self.recipient,
            "type": self.type,
            "confidence": self.confidence,
            "payload": self.payload
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            sender=data.get("from", "Zero"),
            recipient=data.get("to", "ALL"),
            type=data.get("type", "text"),
            confidence=data.get("confidence", 1.0),
            payload=data.get("payload", "")
        )

def get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.jsonl")

def create_session(session_id: str):
    """Initializes a new session chat log."""
    path = get_session_path(session_id)
    if os.path.exists(path):
        return
    
    with open(path, 'w') as f:
        msg = TeamMessage(
            sender="System",
            recipient="ALL",
            type="system",
            payload=f"Session {session_id} initialized."
        )
        f.write(msg.to_json() + "\n")

def log_message(session_id: str, sender: str, payload: Any, type: str = "text", recipient: str = "ALL", confidence: float = 1.0):
    """Appends a message to the session log."""
    path = get_session_path(session_id)
    if not os.path.exists(path):
        create_session(session_id)
    
    msg = TeamMessage(
        sender=sender,
        recipient=recipient,
        type=type,
        confidence=confidence,
        payload=payload
    )
    
    with open(path, 'a') as f:
        f.write(msg.to_json() + "\n")

def read_chat(session_id: str, limit: int = None) -> List[Dict]:
    """Reads messages from the session log."""
    path = get_session_path(session_id)
    if not os.path.exists(path):
        return []
    
    messages = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if limit:
        return messages[-limit:]
    return messages

class TeamRouter:
    """Helper for routing messages between agents."""
    def __init__(self, session_id: str, my_role: str):
        self.session_id = session_id
        self.my_role = my_role
        create_session(session_id)

    def send(self, payload: Any, type: str = "text", to: str = "ALL", confidence: float = 1.0):
        log_message(self.session_id, self.my_role, payload, type, to, confidence)

    def poll(self, last_seen_id: Optional[str] = None) -> List[Dict]:
        msgs = read_chat(self.session_id)
        if not last_seen_id:
            return msgs
        
        # Return only messages after last_seen_id
        new_msgs = []
        found = False
        for m in msgs:
            if found:
                new_msgs.append(m)
            if m["trace_id"] == last_seen_id:
                found = True
        return new_msgs

def main():
    parser = argparse.ArgumentParser(description="Team Runtime CLI")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="Create session")
    create_parser.add_argument("session_id")

    log_parser = subparsers.add_parser("log", help="Log message")
    log_parser.add_argument("session_id")
    log_parser.add_argument("--from", dest="sender", required=True)
    log_parser.add_argument("--payload", required=True)
    log_parser.add_argument("--type", default="text")
    log_parser.add_argument("--to", dest="recipient", default="ALL")
    log_parser.add_argument("--confidence", type=float, default=1.0)

    read_parser = subparsers.add_parser("read", help="Read chat")
    read_parser.add_argument("session_id")
    read_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "create":
        create_session(args.session_id)
        print(f"Created session: {args.session_id}")
    elif args.command == "log":
        log_message(args.session_id, args.sender, args.payload, args.type, args.recipient, args.confidence)
        print("Logged message")
    elif args.command == "read":
        msgs = read_chat(args.session_id, args.limit)
        print(json.dumps(msgs, indent=2))

if __name__ == "__main__":
    main()
