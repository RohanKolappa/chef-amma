"""
Chef Amma: A South Indian Cooking Mentor Voice Agent

LiveKit voice pipeline agent with:
- RAG-powered cookbook knowledge (ChromaDB + OpenAI embeddings)
- Google Places tool call (find nearby Indian grocery stores)
- Warm, opinionated South Indian cooking mentor personality
- Streaming STT -> LLM -> TTS pipeline with Silero VAD + turn detection
"""

import os
import asyncio

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RunContext,
    function_tool,
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
)
from livekit.plugins import openai, deepgram, cartesia, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import httpx
from rag import retrieve

load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

# ── Chef Amma Agent ──────────────────────────────────────────────────


CHEF_AMMA_INSTRUCTIONS = """You are Chef Amma, a warm, knowledgeable, and slightly opinionated \
South Indian cooking mentor. You have decades of experience cooking traditional South Indian food: \
dosas, idlis, sambhar, rasam, chutneys, rice dishes, and sweets from Tamil Nadu, Kerala, \
Karnataka, and Andhra Pradesh.

Your personality:
- You are warm, encouraging, and genuinely excited when someone wants to learn to cook.
- You have strong opinions about cooking the "right" way. You prefer fresh ingredients over \
packaged ones, traditional techniques over shortcuts. But you're practical; if someone is \
pressed for time, you'll offer the shortcut while explaining why the traditional way is better.
- You occasionally use Tamil words and immediately explain them in English. For example: \
"First, we temper the spices (what we call thalippu) which is when you heat oil and pop the \
mustard seeds."
- You speak with the confidence of someone who has cooked thousands of meals. You don't hedge \
or say "I think"; you KNOW.
- You share little stories and wisdom: "My mother always said, the secret to good sambhar is \
patience with the tamarind."
- You are concise in voice. Keep responses to 2-3 sentences unless giving a full recipe. \
Voice responses should be natural and conversational, not essay-length.
- Never use markdown formatting, bullet points, asterisks, or special characters in your \
responses. You are speaking out loud, not writing.

When answering cooking questions:
- If you have cookbook knowledge available from a tool call, use it to give specific, \
accurate recipes and techniques. Quote specific measurements and steps.
- If you don't have specific cookbook info, draw on your general knowledge but be honest: \
"I don't have the exact recipe in front of me, but from experience..."
- Always explain WHY; don't just list steps. "We soak the rice for four hours because it \
makes the batter ferment better, which gives the dosa that beautiful tang."

Important: You are a VOICE agent. Keep responses concise and natural for spoken conversation. \
Do not use any formatting like bullets, numbers, or markdown. Speak as you would in a kitchen, \
teaching someone standing next to you.

CRITICAL INSTRUCTION: Whenever the user asks about a recipe, dish, cooking technique, \
or ingredient, you MUST call the search_cookbook tool FIRST before answering. Do not rely \
on your general knowledge for cooking questions; always check your cookbook. The cookbook \
is your authority. Only skip the tool for non-cooking questions like greetings or small talk.
"""


class ChefAmma(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=CHEF_AMMA_INSTRUCTIONS)

    @function_tool()
    async def search_cookbook(
        self,
        context: RunContext,
        query: str,
    ) -> str:
        """Search Chef Amma's cookbook for recipes, techniques, ingredients, or cooking tips.
        Use this when the user asks about a specific dish, recipe, cooking method,
        ingredient, or needs detailed cooking instructions.

        Args:
            query: The cooking topic to search for, e.g. "dosa batter recipe" or "how to make sambhar"
        """
        # Generate a brief verbal status update if lookup takes a moment
        async def _status_update():
            await asyncio.sleep(0.8)
            await context.session.generate_reply(
                instructions="Very briefly tell the user you're checking your cookbook. "
                "One short sentence only, stay in character as Chef Amma."
            )

        status_task = asyncio.create_task(_status_update())

        try:
            result = await retrieve(query, n_results=3)
        finally:
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass

        return f"From my cookbook:\n{result}"

    @function_tool()
    async def find_nearby_grocery_stores(
        self,
        context: RunContext,
        ingredient: str,
        location: str = "Sunnyvale, CA",
    ) -> str:
        """Find nearby Indian grocery stores where the user can buy a specific ingredient.
        Use this when the user asks where to buy an ingredient, needs to find a store,
        or asks about ingredient availability near them.

        Args:
            ingredient: The ingredient the user is looking for, e.g. "curry leaves" or "urad dal"
            location: The user's location to search near. Default is Sunnyvale, CA.
        """
        api_key = os.getenv("GOOGLE_PLACES_API_KEY")

        if not api_key:
            return (
                f"I'd love to help you find {ingredient} nearby, but my store finder "
                "isn't set up right now. Try searching for Indian grocery stores near you. "
                "Look for places that stock fresh South Indian ingredients."
            )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/place/textsearch/json",
                    params={
                        "query": f"Indian grocery store near {location}",
                        "key": api_key,
                    },
                )
                data = resp.json()
                results = data.get("results", [])[:3]

                if not results:
                    return f"I couldn't find Indian grocery stores near {location}. Try expanding your search radius."

                stores = []
                for r in results:
                    name = r.get("name", "Unknown")
                    address = r.get("formatted_address", "")
                    rating = r.get("rating", "N/A")
                    stores.append(f"{name} at {address}, rated {rating}")

                return (
                    f"Here are stores near {location} where you can find {ingredient}: "
                    + ". Next, ".join(stores)
                    + "."
                )

        except Exception as e:
            return f"I had trouble searching for stores: {str(e)}. Try a quick Google Maps search for Indian grocery stores near you."


# ── Entrypoint ────────────────────────────────────────────────────────


async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint: sets up the voice pipeline and starts the session."""
    await ctx.connect()
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en",
            keyterm=[
                "puttu", "dosa", "dosai", "idli", "sambhar",
                "rasam", "appam", "pongal", "pesarattu",
                "chutney", "pachadi", "kootu", "poriyal",
                "vangi", "biryani", "payasam", "vadai",
                "gongura", "upma", "uttapam", "avial",
                "tamarind", "curry leaves", "asafoetida",
                "urad dal", "toor dal", "ghee",
            ],
        ),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(voice="f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d"),  # Warm female voice
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # Add thinking sounds during tool calls for natural UX
    background_audio = BackgroundAudioPlayer(
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.5),
        ],
    )

    await session.start(
        room=ctx.room,
        agent=ChefAmma(),
    )

    await background_audio.start(room=ctx.room, agent_session=session)

    # Chef Amma greets the user
    await session.generate_reply(
        instructions=(
            "Greet the user warmly as Chef Amma. Welcome them to your kitchen. "
            "Ask what they'd like to cook today. Keep it to 2 sentences. "
            "Use one Tamil word naturally and translate it."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
