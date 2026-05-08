import os
import uuid
import json
from http.server import BaseHTTPRequestHandler
from livekit.api import AccessToken, VideoGrants

LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET or not LIVEKIT_URL:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing LiveKit environment variables"}).encode())
            return
        room_name = f"chef-amma-{uuid.uuid4().hex[:8]}"
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

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

        body = json.dumps({
            "token": token.to_jwt(),
            "room": room_name,
            "identity": user_identity,
            "livekit_url": LIVEKIT_URL,
        })
        self.wfile.write(body.encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
