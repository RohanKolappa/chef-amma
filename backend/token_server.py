"""
Token server for Chef Amma voice agent.

Generates LiveKit room tokens so the React frontend can join a room.
The agent (running separately) auto-dispatches to any room that needs one.

Run:
    uvicorn token_server:app --port 8080 --reload

How room tokens work:
    A room token is a JWT (JSON Web Token) signed with your LiveKit API secret.
    It contains:
    - identity: who the participant is (e.g., "user-abc123")
    - grants: what they can do (join room, publish audio, subscribe to tracks)
    - room: which LiveKit room to join

    The frontend receives this JWT, passes it to the LiveKit React SDK,
    and the WebRTC connection is established automatically.
"""

import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants

load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")

if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL]):
    import warnings
    warnings.warn("Missing LiveKit environment variables — token generation will fail")

app = FastAPI(title="Chef Amma Token Server")

# Allow the React frontend (running on localhost:5173) to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/token")
async def get_token(room: str | None = None):
    """
    Generate a LiveKit room token for the frontend.

    The token grants the user permission to join the specified room,
    publish their microphone audio, and subscribe to the agent's audio.
    """
    room_name = room or f"chef-amma-{uuid.uuid4().hex[:8]}"
    user_identity = f"user-{uuid.uuid4().hex[:6]}"

    token = AccessToken(
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
    )
    token.identity = user_identity
    token.name = "Chef Amma User"
    token.with_grants(
        VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        )
    )

    return {
        "token": token.to_jwt(),
        "room": room_name,
        "identity": user_identity,
        "livekit_url": LIVEKIT_URL,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "agent": "chef-amma"}
