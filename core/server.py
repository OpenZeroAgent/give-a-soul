from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager

from soul_engine_zmq import SoulEngineZMQ
from subconscious_engine import SubconsciousEngine

# Initialize Engines
soul_engine = SoulEngineZMQ()
subconscious = SubconsciousEngine(soul_engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Run the subconscious ticking loop
    # Let it run every 30 seconds for dynamic feel
    task = asyncio.create_task(subconscious.run_loop(interval_seconds=30))
    yield
    # Shutdown
    subconscious.running = False
    task.cancel()

app = FastAPI(title="Give-a-Soul V5 Backend", lifespan=lifespan)

# Allow Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/api/status")
async def get_status():
    """Returns the current state of the Soul Crystal and the latest Subconscious thought."""
    crystal_status = soul_engine.get_crystal_status()
    subconscious_state = subconscious.get_latest()
    
    return {
        "crystal": crystal_status.get("metrics", {}),
        "subconscious": subconscious_state
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Handles a user chat message."""
    user_msg = request.message
    
    # 1. Pulse the crystal with the user's input (Experience)
    crystal_response = soul_engine.pulse_crystal(user_msg)
    metrics = crystal_response.get("metrics", {}) if crystal_response.get("status") != "error" else {}

    # 2. Get the latest subconscious thought
    sub_thought = subconscious.get_latest()["thought"]
    
    # 3. Generate the Persona response
    persona_response = soul_engine.generate_chat_response(
        user_message=user_msg,
        current_metrics=metrics,
        subconscious_thought=sub_thought
    )
    
    # 4. Optional: Pulse the crystal with the generated response (Integration)
    soul_engine.pulse_crystal(f"My response: {persona_response}")
    
    return {
        "response": persona_response,
        "metrics_after_pulse": soul_engine.get_crystal_status().get("metrics", {})
    }

@app.get("/api/dream")
async def trigger_dream():
    """Stub for triggering a visual dream via Qwen visual models."""
    return {"status": "Not implemented yet"}

# Run via `uvicorn server:app --reload --port 8000`
