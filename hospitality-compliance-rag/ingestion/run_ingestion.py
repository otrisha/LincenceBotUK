"""
ingestion/run_ingestion.py
───────────────────────────
Single entry point for the offline ingestion pipeline.

Run once to populate:
  1. retrieval/bm25_index.pkl  — serialised BM25Index for sparse retrieval
  2. Pinecone index            — dense vectors for semantic retrieval

Usage:
    pip install -r requirements_ingestion.txt
    python -m ingestion.run_ingestion

⚠️  This script MUST NEVER be imported by the Streamlit app or any
    runtime module. It is a standalone offline process only.

⚠️  Requires a .env file (copy .env.example → .env and fill in keys).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Load .env before anything else so all modules read env vars correctly
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from ingestion.document_loader import load_all_documents
from ingestion.chunker import chunk_all_documents, Chunk
from ingestion.embedder import EmbeddingModel

# BM25 — built and serialised here
import pickle
import nltk
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", ROOT / "retrieval" / "bm25_index.pkl"))

PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hospitality-compliance")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "hospitality-kb")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION  = int(os.getenv("PINECONE_DIMENSION", 1024))
PINECONE_METRIC     = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_BATCH_SIZE = int(os.getenv("PINECONE_UPSERT_BATCH", 100))

BM25_K1 = float(os.getenv("BM25_K1", 1.5))
BM25_B  = float(os.getenv("BM25_B", 0.75))


# ---------------------------------------------------------------------------
# NLTK data (tokenisation)
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            print(f"  Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_STOPWORDS = set()

def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if not _STOPWORDS:
        from nltk.corpus import stopwords as sw
        _STOPWORDS = set(sw.words("english"))
    return _STOPWORDS


def _tokenise(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    stops = _get_stopwords()
    return [t for t in tokens if t.isalnum() and t not in stops]


class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi together with the ordered chunk list.

    The chunk list and the BM25 corpus share the same positional index —
    score at position i corresponds to chunks[i].
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        print(f"  Building BM25 index over {len(chunks)} chunks ...")
        corpus = [_tokenise(c.bm25_text) for c in chunks]
        self._bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
        print("  BM25 index built.")

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  BM25 index saved → {path}")

    @classmethod
    def load(cls, path: Path = BM25_INDEX_PATH) -> "BM25Index":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        tokens = _tokenise(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k] if score > 0]

    def get_chunk(self, index: int) -> Chunk:
        return self.chunks[index]

    def __len__(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Pinecone upsert
# ---------------------------------------------------------------------------

def _ensure_pinecone_index(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"  Creating Pinecone index: {PINECONE_INDEX_NAME} ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print("  Waiting for index to be ready ...")
        for _ in range(30):
            status = pc.describe_index(PINECONE_INDEX_NAME).status
            if status.get("ready"):
                break
            time.sleep(2)
        else:
            raise RuntimeError("Pinecone index did not become ready in 60 s.")
    print(f"  Pinecone index ready: {PINECONE_INDEX_NAME}")


def upsert_to_pinecone(chunks: list[Chunk], vectors) -> None:
    """Upload all chunk vectors and metadata to Pinecone."""
    import numpy as np
    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_pinecone_index(pc)
    index = pc.Index(PINECONE_INDEX_NAME)

    records = []
    for chunk, vec in zip(chunks, vectors):
        records.append({
            "id": chunk.chunk_id,
            "values": vec.tolist(),
            "metadata": {
                "source_document": chunk.source_document,
                "heading":         chunk.heading[:200],
                "section_number":  chunk.section_number or "",
                "page_number":     chunk.page_number,
                "chunk_index":     chunk.chunk_index,
                "topic_category":  chunk.topic_category,
                "chunking_method": chunk.chunking_method,
                "token_count":     chunk.token_count,
                # text is intentionally NOT stored in Pinecone;
                # it is looked up from the BM25 index by chunk_id at runtime.
            },
        })

    total = len(records)
    print(f"  Upserting {total} vectors to Pinecone in batches of {PINECONE_BATCH_SIZE} ...")
    for start in range(0, total, PINECONE_BATCH_SIZE):
        batch = records[start : start + PINECONE_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
        print(f"    Upserted {min(start + PINECONE_BATCH_SIZE, total)}/{total}")
    print("  Pinecone upsert complete.")


# ---------------------------------------------------------------------------
# Chunk serialisation (JSON snapshot for inspection)
# ---------------------------------------------------------------------------

def _save_chunks_json(chunks: list[Chunk]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "all_chunks.json"
    data = [
        {
            "chunk_id":       c.chunk_id,
            "source_document": c.source_document,
            "heading":        c.heading,
            "section_number": c.section_number,
            "page_number":    c.page_number,
            "chunk_index":    c.chunk_index,
            "token_count":    c.token_count,
            "topic_category": c.topic_category,
            "chunking_method": c.chunking_method,
            "text":           c.text,
        }
        for c in chunks
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Chunk snapshot saved → {out_path} ({len(data)} chunks)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ingestion() -> None:
    print("\n" + "=" * 60)
    print("  UK Hospitality Compliance RAG — Ingestion Pipeline")
    print("=" * 60)

    # 1. NLTK resources
    print("\n[1/5] Ensuring NLTK resources ...")
    _ensure_nltk()

    # 2. Load PDFs
    print(f"\n[2/5] Loading documents from {RAW_DIR} ...")
    documents = load_all_documents(RAW_DIR)
    print(f"  Total documents loaded: {len(documents)}")

    # 3. Chunk
    print("\n[3/5] Chunking documents ...")
    chunks = chunk_all_documents(documents)
    print(f"  Total chunks produced: {len(chunks)}")
    _save_chunks_json(chunks)

    # 4. BM25 index
    print("\n[4/5] Building and serialising BM25 index ...")
    bm25 = BM25Index(chunks)
    bm25.save(BM25_INDEX_PATH)

    # 5. Embed + Pinecone
    print("\n[5/5] Encoding chunks and upserting to Pinecone ...")
    vectors = EmbeddingModel.encode_chunks(chunks)
    upsert_to_pinecone(chunks, vectors)

    print("\n" + "=" * 60)
    print("  Ingestion complete.")
    print(f"  BM25 index : {BM25_INDEX_PATH}")
    print(f"  Pinecone   : {PINECONE_INDEX_NAME} / {PINECONE_NAMESPACE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_ingestion()
