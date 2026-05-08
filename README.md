# Chef Amma: South Indian Cooking Mentor Voice Agent

A RAG-enabled voice agent built with LiveKit that serves as a warm, opinionated South Indian cooking mentor. Ask her about dosas, sambhar, rasam, or any South Indian dish; she'll guide you through recipes from her cookbook, share cooking wisdom, and help you find Indian grocery stores near you.

**Live Demo:** [https://chef-amma.vercel.app](https://chef-amma.vercel.app) (requires the agent to be running locally; see [Deployment](#deployment))

## Why Chef Amma?

I love cooking, and I have family in Chennai. I tend to call my own Amma (Tamil word for Mom) and ask her for tips and tricks on how to make Indian food when I'm away from home, so I thought that I could build a tool/mentor that could help me when she's not free. Chef Amma is that mentor: she has strong opinions about doing things properly, sprinkles in Tamil words naturally, and genuinely gets excited when you want to learn a new dish. Just like my own Amma.

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│              React Frontend (Vite + TS)              │
│   Start Call ─── Live Transcript ─── End Call        │
│   @livekit/components-react (WebRTC transport)       │
└──────────────┬──────────────────────────────────────┘
               │ Room Token (JWT)
┌──────────────▼──────────────┐
│       Token Server           │◄── generates room tokens
│       GET /api/token         │    signed with LIVEKIT_API_SECRET
└──────────────────────────────┘

               │ WebRTC via LiveKit Cloud
┌──────────────▼──────────────────────────────────────┐
│              LiveKit Agent (Python)                   │
│                                                       │
│  Audio In ──► Deepgram STT ──► OpenAI GPT-4o ──►    │
│               (Nova-3)          (streaming)           │
│                                     │                 │
│                              Tool Calls?              │
│                           ┌─────┴─────┐              │
│                           │           │              │
│                    search_cookbook  find_stores         │
│                           │           │              │
│                    ChromaDB       Google Places        │
│                    (cosine)       (Text Search)        │
│                           │           │              │
│                           └─────┬─────┘              │
│                                 ▼                     │
│              ◄── Cartesia TTS ◄─┘                    │
│                  (Sonic, streaming)                    │
│                                                       │
│  VAD: Silero │ Turn Detection: Multilingual Model     │
└──────────────────────────────────────────────────────┘
```

The token server has two implementations: a FastAPI server (`backend/token_server.py`) for local development, and a Python serverless function (`api/token.py`) for the Vercel deployment. Both generate JWTs with the same grants via GET /api/token. The local FastAPI version additionally accepts an optional room query parameter and exposes a GET /api/health endpoint for debugging.

### Voice Pipeline (STT -> LLM -> TTS)

The agent uses a **cascaded streaming pipeline** where all three stages overlap:

- **STT (Deepgram Nova-3):** Streams partial transcripts as the user speaks. Chosen for low-latency streaming and strong conversational speech accuracy. Configured with keyterm prompting for South Indian food vocabulary (puttu, dosa, sambhar, pesarattu, etc.) to improve transcription accuracy for domain-specific terms that would otherwise be misheard as common English words.
- **LLM (OpenAI GPT-4o):** Receives finalized transcript, generates response with streaming tokens. Handles tool calling for RAG lookups and store searches.
- **TTS (Cartesia Sonic):** Begins synthesizing audio as soon as it receives the first sentence from the LLM. Low time-to-first-audio-byte (~100ms).

Streaming at every stage keeps mouth-to-ear latency under ~800ms for non-tool-call turns.

### Turn Detection & VAD

- **Silero VAD** detects speech presence (is someone speaking right now?)
- **LiveKit Multilingual Turn Detector** determines if the user is *done* speaking — it analyzes semantic completeness of the transcribed text, not just silence duration. This prevents the agent from interrupting mid-thought when the user pauses to think.

### Room Token Generation

The token server generates JWTs signed with the LiveKit API secret via a `GET /api/token` endpoint. Each token contains a user identity, room name, and grants (`room_join`, `can_publish`, `can_subscribe`). The frontend receives this JWT and passes it to the LiveKit React SDK, which establishes a WebRTC connection automatically.

For local development, the token server runs as a FastAPI app (`backend/token_server.py`). For the Vercel deployment, it runs as a Python serverless function (`api/token.py`) using `BaseHTTPRequestHandler`.

## RAG Integration

### Ingestion Pipeline

```
Cookbook PDF ──► PyMuPDF text extraction ──► Sliding window chunking ──►
OpenAI embeddings (text-embedding-3-small) ──► ChromaDB (persistent, local)
```

**Chunking strategy:** Fixed-size sliding window, 500 characters with 100 character overlap.

**Why this approach:**
- Simple, predictable chunk sizes
- Overlap ensures no sentence is fully lost at chunk boundaries
- Works well for prose-heavy cookbook content (recipes, technique descriptions)

**Trade-off:** This cuts across semantic boundaries: a recipe might start in one chunk and end in the next. In production, I'd implement **recipe-level chunking** where each recipe becomes a single chunk, preserving the full context of ingredients + steps + tips. This requires parsing document structure (detecting recipe headers/boundaries), which adds complexity.

### Retrieval

When the LLM calls the `search_cookbook` tool:

1. Query text is embedded using `text-embedding-3-small`
2. ChromaDB returns the top 3 nearest chunks by cosine similarity
3. Results are filtered by a relevance threshold (cosine distance < 0.8) to avoid injecting noise
4. Relevant chunks are returned to the LLM as tool call output

**Embedding model choice:** `text-embedding-3-small` (1536 dimensions). For a cooking domain with distinct vocabulary (ingredient names, technique terms, dish names), the smaller model retrieves accurately. `text-embedding-3-large` would give marginal improvement at 6.5x cost.

**Top-k = 3:** Enough context for accurate answers without overwhelming the LLM's context window or adding unnecessary latency. Each additional chunk adds ~20-50ms to LLM processing time, which matters in a voice context.

### RAG Injection: Tool Call vs. Auto-Inject

I chose **RAG as a tool call** (the LLM decides when to search) rather than auto-injecting context on every turn via `on_user_turn_completed`.

**Why tool call:**
- Only searches when the LLM judges it's needed; it avoids wasting tokens on "how are you?" or non-cooking questions
- The LLM can reformulate the query for better retrieval (the raw speech transcript can be noisy)
- Explicit and debuggable: you can see exactly when and why the search happened

**The trade-off:** Tool calls add a full round-trip of latency (LLM decides -> tool executes -> LLM generates). This adds ~500-800ms to the response. I mitigate this with:
- A verbal status update ("Let me check my cookbook...") if the lookup takes > 0.8s
- Background "thinking" sounds via LiveKit's BackgroundAudioPlayer

**If I had more time:** I'd implement a **hybrid approach**: use `on_user_turn_completed` with a relevance score threshold. Auto-inject context only when the top result's cosine similarity exceeds 0.85, skip injection otherwise. This gives the latency benefit of auto-inject with less noise than always injecting.

## Tool Calls

### `search_cookbook(query)`
Searches the vector store for recipes, techniques, and ingredient info. Returns concatenated top-k chunks as context for the LLM. Includes a delayed status update pattern: if the lookup takes longer than 0.8s, the agent speaks a brief filler message to maintain conversational flow during the wait.

### `find_nearby_grocery_stores(ingredient, location)`
Uses Google Places Text Search API to find Indian grocery stores near the user's location. Returns store names, addresses, and ratings. The default location is Sunnyvale, CA, but the agent can pick up location changes from conversation when GPT-4o infers the location from context and passes it to the tool call (e.g., "I'm in Oahu" triggers a search in Oahu, HI). Gracefully degrades if the API key isn't configured.

## Testing & Evaluation

The project includes a retrieval quality test suite (`backend/test_rag.py`) that validates the RAG pipeline against known ground truth from the cookbook. The suite runs 11 test cases across recipe, ingredient, technique, and edge_case categories, with expected keywords derived from specific cookbook pages, and verifies that the retrieved chunks contain the correct content.

This is a seed implementation of what a comprehensive evaluation framework would look like. In production, I'd extend it along several dimensions:

- **Retrieval evaluation:** Expand beyond keyword matching to use LLM-as-judge scoring for semantic accuracy, measuring whether retrieved chunks actually answer the question asked.
- **End-to-end voice evaluation:** Test the full pipeline from speech input to spoken response: verifying STT accuracy on domain-specific terms, tool call triggering rates (does the agent call `search_cookbook` when it should?), and response grounding (does the agent cite cookbook content vs. hallucinating from parametric knowledge?).
- **Regression detection:** Run the test suite automatically after any change to chunking parameters, embedding models, or system prompts to catch quality regressions before they reach users.
- **Simulated conversations:** Generate synthetic user interactions that cover edge cases (ambiguous queries, follow-up questions, non-cooking topics) and evaluate agent behavior against rubrics. This is the direction automated QA platforms take: defining expected behavior, simulating diverse user patterns, and scoring results at scale.

## Deployment

- **Frontend + Token Server:** Deployed on Vercel at [https://chef-amma.vercel.app](https://chef-amma.vercel.app). The React frontend is built as a static site. The token server runs as a Vercel Python serverless function (`api/token.py`).
- **Agent:** Runs locally, connecting to LiveKit Cloud via WebSocket. The agent auto-dispatches to any room created through the deployed frontend. For the agent to respond, it must be running locally with `python agent.py dev`.
- **LiveKit Cloud:** Handles all WebRTC media routing. Chosen for managed infrastructure; no need to self-host a media server. Free tier is sufficient for development and demos.
- **ChromaDB:** Runs locally as a persistent vector store. In production, I'd use a hosted solution like Pinecone or pgvector for reliability and horizontal scaling.

## Running Locally

### Prerequisites
- Python 3.10-3.13 (3.14 is not yet supported by LiveKit)
- Node.js 18+
- LiveKit Cloud account (free tier works): https://cloud.livekit.io
- OpenAI API key
- Deepgram API key (free tier: $200 credit): https://console.deepgram.com
- Cartesia API key (free tier available): https://play.cartesia.ai
- Google Places API key (optional, for store finder)

### Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/RohanKolappa/chef-amma.git
cd chef-amma

# 2. Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
cp .env.example .env.local
# Edit .env.local with your API keys

pip install -r requirements.txt

# 3. Download required model files (turn detection + VAD)
python agent.py download-files

# 4. Ingest the cookbook PDF
python ingest.py ../data/cookbook.pdf

# 5. Start the token server (Terminal 1)
uvicorn token_server:app --port 8080 --reload

# 6. Start the agent (Terminal 2)
python agent.py dev

# 7. Frontend setup (Terminal 3)
cd ../frontend
npm install
npm run dev
```

Open http://localhost:5173 and click "Start Cooking Session."

## What I'd Improve

- **Recipe-level chunking** for the RAG pipeline, preserving full recipe context
- **Hybrid RAG injection** with relevance thresholds to reduce tool call latency
- **Conversation memory** so Chef Amma remembers what you've already asked about in the session
- **Production vector store** (Pinecone or pgvector) instead of local ChromaDB
- **Voice selection A/B testing** to find the TTS voice that best matches the persona
- **Comprehensive evaluation suite** — expand `test_rag.py` into a full pipeline evaluation framework covering retrieval quality, STT accuracy, tool call behavior, and response grounding

## AI Tools Used

- Claude (Anthropic) for project scaffolding, architecture planning, code generation, and debugging during development