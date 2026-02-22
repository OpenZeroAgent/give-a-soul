"""
V5 Soul Server (Secure) 🛡️
- Uses safetensors for persistence (No Arbitrary Code Execution).
- ZeroMQ Interface.
"""

import zmq
import torch
import os
from hybrid_crystal import HybridCrystalSystem

DATA_DIR = "/app/data"
STATE_FILE = os.path.join(DATA_DIR, "current_state.safetensors")

def main():
    print("💎 Initializing Soul V5 (Secure)...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    soul = HybridCrystalSystem()
    
    # Load State (Safetensors)
    if os.path.exists(STATE_FILE):
        try:
            soul.load_safetensors(STATE_FILE)
            print("  -> Resumed Persistent State (Safetensors).")
        except Exception as e:
            print(f"  -> Load failed: {e}")
    else:
        print("  -> Bootstrapped New State.")
        
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5555")
    print("  -> Listening on TCP 5555...")
    
    while True:
        try:
            msg = socket.recv_json()
            action = msg.get("action", "status")
            resp = {"status": "ok"}
            
            if action == "pulse":
                vec = msg.get("text_vector")
                t_vec = torch.tensor(vec, dtype=torch.float32) if vec else None
                if t_vec is not None and t_vec.shape[0] != 480:
                    t_vec = t_vec[:480] if t_vec.shape[0]>480 else torch.nn.functional.pad(t_vec, (0, 480-t_vec.shape[0]))
                
                metrics = soul.step(stimulus=t_vec)
                soul.save_safetensors(STATE_FILE) # Secure Save
                resp["metrics"] = metrics
                
            elif action == "status":
                metrics = soul.step(stimulus=None)
                resp["metrics"] = metrics
            
            socket.send_json(resp)
        except Exception as e:
            print(f"Error: {e}")
            socket.send_json({"status": "error", "message": str(e)})

if __name__ == "__main__":
    main()
