"""
RAG module for Chef Amma voice agent.

Handles embedding generation and vector store retrieval
against the ingested cookbook PDF.
"""

import chromadb
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

# ── Clients ──────────────────────────────────────────────────────────
_openai_client = AsyncOpenAI()
_chroma_client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(__file__), "chroma_db"))

COLLECTION_NAME = "cookbook"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_collection():
    """Get or create the cookbook collection."""
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


async def embed_text(text: str) -> list[float]:
    """Generate an embedding for a single text string."""
    response = await _openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


async def retrieve(query: str, n_results: int = 3) -> str:
    """
    Retrieve the top-k most relevant cookbook chunks for a query.

    Returns the concatenated text of the top chunks, or a fallback
    message if the collection is empty or results are poor.
    """
    collection = get_collection()

    if collection.count() == 0:
        return "No cookbook data has been ingested yet."

    query_embedding = await embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "distances"],
    )

    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    if not documents:
        return "I couldn't find anything relevant in my cookbook."

    # Filter out low-relevance results (cosine distance > 0.8 means low similarity)
    relevant = [
        doc for doc, dist in zip(documents, distances)
        if dist < 0.8
    ]

    if not relevant:
        return "I couldn't find a strong match in my cookbook for that topic."

    return "\n\n---\n\n".join(relevant)
