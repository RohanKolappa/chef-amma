"""
Ingestion script for Chef Amma's cookbook RAG pipeline.

Extracts text from a PDF, chunks it with a sliding window,
generates embeddings via OpenAI, and stores them in ChromaDB.

Usage:
    python ingest.py <path_to_cookbook_pdf>

Trade-off notes:
- Chunking: Fixed-size sliding window (500 chars, 100 char overlap).
  Simple and predictable. In production, I'd use recipe-level chunking
  (each recipe as one unit) to preserve semantic boundaries, but that
  requires parsing document structure which adds complexity for a take-home.
- Overlap: 100 chars ensures no sentence is split without representation
  in at least one chunk. Larger overlap = more redundancy but better recall.
- Embedding model: text-embedding-3-small is fast and cheap. For a cooking
  domain with distinct vocabulary (ingredient names, technique terms), the
  smaller model retrieves accurately. text-embedding-3-large would give
  marginal improvement at 6.5x the cost.
"""

import sys
import os
import fitz  # PyMuPDF
from openai import OpenAI
import chromadb

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

# ── Config ────────────────────────────────────────────────────────────
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "cookbook"
BATCH_SIZE = 50        # embed this many chunks per API call


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF, page by page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()

    full_text = "\n\n".join(pages)
    print(f"  Extracted {len(full_text):,} characters from {len(pages)} pages")
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks using a sliding window.

    This is the simplest chunking strategy. Trade-offs:
    - Pro: Predictable chunk sizes, simple to implement
    - Con: Can split mid-sentence or mid-recipe
    - Alternative: Recursive splitting (by paragraph -> sentence -> char)
      preserves semantic boundaries but produces uneven chunk sizes
    - Alternative: Recipe-level chunking (best for cookbooks, but requires
      parsing the document structure to identify recipe boundaries)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        # Skip very short chunks (likely just whitespace or page breaks)
        if len(chunk) > 50:
            chunks.append(chunk)

        start += chunk_size - overlap

    print(f"  Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def embed_and_store(chunks: list[str]):
    """Generate embeddings and store chunks in ChromaDB."""
    client = OpenAI()
    chroma_client = chromadb.PersistentClient(
        path=os.path.join(os.path.dirname(__file__), "chroma_db")
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Clear existing data for clean re-ingestion
    if collection.count() > 0:
        print(f"  Clearing {collection.count()} existing chunks...")
        collection.delete(where={"source": "cookbook"})

    # Process in batches to avoid API rate limits
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        # Generate embeddings for the batch
        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
        )
        embeddings = [item.embedding for item in response.data]

        # Store in ChromaDB
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        metadatas = [{"source": "cookbook", "chunk_index": i + j} for j in range(len(batch))]

        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        print(f"  Embedded and stored chunks {i+1}-{i+len(batch)} of {len(chunks)}")

    print(f"\n  Done! {collection.count()} chunks in collection '{COLLECTION_NAME}'")


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_cookbook_pdf>")
        print("Example: python ingest.py ../data/cookbook.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"\nIngesting cookbook: {pdf_path}\n")

    print("Step 1: Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("\nStep 2: Chunking text...")
    chunks = chunk_text(text)

    print("\nStep 3: Embedding and storing in ChromaDB...")
    embed_and_store(chunks)

    print("\nIngestion complete! Your cookbook is ready for Chef Amma.\n")


if __name__ == "__main__":
    main()
