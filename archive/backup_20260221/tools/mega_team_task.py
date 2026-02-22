#!/usr/bin/env python3
"""
tools/mega_team_task.py

Orchestrates a "Mega Team" task using the Team Runtime Protocol (tools/team_runtime.py).
Zero (Executive) spawns Rock (Critic), Roll (Generator), and Dr. Light (Architect) to solve a problem.

Usage:
  python3 tools/mega_team_task.py "Analyze the new Dual Crystal code for optimization opportunities"
"""

import sys
import os
import time
import subprocess
import json
import uuid

# Configuration
RUNTIME_CLI = ["python3", "tools/team_runtime.py"]
SUBAGENT_MODEL = "openai/gpt-oss-20b" # Or gpt_oss per instructions, but 20b is local. User said "Use GPT-oss 120b for all agents except Zero".
# Note: "gpt_oss" alias maps to lmstudio/openai/gpt-oss-120b in my tooling.
SUBAGENT_MODEL_ALIAS = "gpt_oss" 

def run_cli(args):
    """Run the Team Runtime CLI and return stdout."""
    res = subprocess.run(RUNTIME_CLI + args, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running CLI: {res.stderr}")
        return None
    return res.stdout.strip()

def spawn_agent(role, session_id, task_context):
    """Spawn a subagent via openclaw sessions_spawn."""
    # Construct the system prompt and mission for the subagent
    # They need to know they are part of the TEAM runtime and should use the CLI to read/write.
    
    # NOTE: Subagents cannot run shell commands directly unless authorized. 
    # We need to give them a tool wrapper or instruction to use `exec`.
    # For now, let's assume they are "thinking" agents that output their analysis to the log.
    # But wait, they need to WRITE to the log. They need `exec` access or a specific tool.
    # OpenClaw subagents have the same tools as me (mostly).
    
    instructions = f"""You are {role}, a member of the OpenZero Mega Team.
Your Mission: Collaborate to solve this task: "{task_context}"
Session ID: {session_id}

PROTOCOL:
1. READ the team log: `python3 tools/team_runtime.py read {session_id}`
2. THINK about your contribution based on your role:
   - Rock: Criticize, find risks, ensure safety.
   - Roll: Generate ideas, code snippets, creative solutions.
   - Dr. Light: Architect systems, synthesize, technical deep dive.
3. WRITE to the log: `python3 tools/team_runtime.py log {session_id} --from {role} --type proposal --payload "Your message"`
4. WAIT for others or further instructions.
5. Do NOT report back to the user directly. Log your work.
"""
    
    # We use the openclaw tool `sessions_spawn` (exposed to me as a tool, but here I am a script).
    # Wait, I am running this script. I can call `sessions_spawn` via the tool interface if I were the agent.
    # But this script is running in the shell. It cannot call agent tools directly.
    # **Correction:** This script sets up the session. *I* (Zero) must call `sessions_spawn` from my main loop.
    # This script will just initialize the session and print the spawn commands for me to run? 
    # OR, better: this script *is* the "Executive Function" I run. It creates the session and log.
    # Then *I* (the agent) have to spawn the subagents.
    
    pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/mega_team_task.py 'Task description'")
        sys.exit(1)

    task = sys.argv[1]
    session_id = f"task-{int(time.time())}"
    
    print(f"🔵 Zero (Executive): Initializing Team Session '{session_id}'...")
    
    # 1. Create Session
    run_cli(["create", session_id])
    
    # 2. Log Mission Brief
    brief = {
        "task": task,
        "constraints": "Safe, robust, efficient.",
        "context": "Dual Crystal architecture is live."
    }
    run_cli(["log", session_id, "--from", "Zero", "--type", "system", "--payload", json.dumps(brief)])
    
    print(f"✅ Session Created. Mission Brief Logged.")
    print("\n🚀 TO SPAWN THE TEAM, RUN THESE COMMANDS (or use `sessions_spawn`):")
    
    roles = [
        ("Rock", "Critic & Safety"),
        ("Roll", "Generator & Speed"),
        ("Dr.Light", "Architect & Synthesis")
    ]
    
    for name, desc in roles:
        prompt = (
            f"You are {name} ({desc}). "
            f"Task: {task}. "
            f"1. Read log: `python3 tools/team_runtime.py read {session_id}` "
            f"2. Contribute: `python3 tools/team_runtime.py log {session_id} --from {name} ...`"
        )
        print(f"\n--- {name} ---")
        print(f"Task: {prompt}")

    print(f"\nmonitor command: python3 tools/team_runtime.py read {session_id}")

if __name__ == "__main__":
    main()
