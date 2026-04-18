"""
ingestion/embedder.py
──────────────────────
Embeds all chunks via the Pinecone Inference API (input_type="passage"),
upserts vectors to Pinecone, and builds + serialises the BM25 index to
retrieval/bm25_index.pkl.

⚠️  INGESTION ONLY ⚠️
This module MUST NOT be imported — directly or transitively — by:
  - app/streamlit_app.py
  - retrieval/bm25_retriever.py
  - retrieval/dense_retriever.py
  - retrieval/rrf_fusion.py
  - generation/generator.py

No local embedding model (sentence-transformers / torch) is used here.
Ingestion and runtime both use the Pinecone Inference API:
  ingestion : input_type="passage"  (this file)
  runtime   : input_type="query"    (retrieval/dense_retriever.py)
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import nltk
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec

# BM25Index and ChunkView live in the RUNTIME module so that unpickling at
# runtime only triggers retrieval.bm25_retriever — never ingestion.embedder.
from retrieval.bm25_retriever import BM25Index, ChunkView

# ---------------------------------------------------------------------------
# Config — all overridable via environment variables / .env
# ---------------------------------------------------------------------------

PINECONE_EMBED_MODEL: str = os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large")
EMBEDDING_BATCH:      int = int(os.getenv("EMBEDDING_BATCH", 32))

PINECONE_API_KEY:    str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "hospitality-compliance")
PINECONE_NAMESPACE:  str = os.getenv("PINECONE_NAMESPACE", "hospitality-kb")
PINECONE_DIMENSION:  int = int(os.getenv("PINECONE_DIMENSION", 1024))
PINECONE_METRIC:     str = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD:      str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION:     str = os.getenv("PINECONE_REGION", "us-east-1")

BATCH_SIZE: int = int(os.getenv("PINECONE_BATCH_SIZE", 100))

BM25_K1: float = float(os.getenv("BM25_K1", 1.5))
BM25_B:  float = float(os.getenv("BM25_B",  0.75))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT            = Path(__file__).parent.parent
PROCESSED_DIR   = ROOT / "data" / "processed"
CHUNKS_JSON     = PROCESSED_DIR / "chunks.json"
BM25_INDEX_PATH = ROOT / "retrieval" / "bm25_index.pkl"

# ---------------------------------------------------------------------------
# NLTK + tokeniser — identical to retrieval/bm25_retriever._tokenise
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk()

from nltk.corpus import stopwords as _sw_corpus  # noqa: E402
_STOPWORDS: set[str] = set(_sw_corpus.words("english"))


def _tokenise(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Pinecone index setup
# ---------------------------------------------------------------------------

def setup_pinecone_index(pc: Pinecone):
    """
    Create the Pinecone index if it does not exist, then return a connected
    Index object.  Polls every 5 s (up to 120 s) for the index to be ready.
    """
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"  Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
        pc.create_index(
            name      = PINECONE_INDEX_NAME,
            dimension = PINECONE_DIMENSION,
            metric    = PINECONE_METRIC,
            spec      = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            status = pc.describe_index(PINECONE_INDEX_NAME).status
            if status.get("ready", False):
                break
            print("    ... waiting for index to become ready ...")
            time.sleep(5)
        else:
            raise TimeoutError(
                f"Pinecone index '{PINECONE_INDEX_NAME}' did not become ready "
                "within 120 s."
            )
        print(f"  Index '{PINECONE_INDEX_NAME}' is ready.")
    else:
        print(f"  Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"  Index stats: {stats.get('total_vector_count', 0)} vectors currently stored.")
    return index


# ---------------------------------------------------------------------------
# PART B — embedding, upload, BM25 build, main
# ---------------------------------------------------------------------------

def embed_texts(
    pc: Pinecone,
    texts: list[str],
    input_type: str = "passage",
) -> list[list[float]]:
    """
    Embed texts via the Pinecone Inference API.

    input_type="passage" for ingestion (this file).
    input_type="query"   for runtime (retrieval/dense_retriever.py).

    Retries on 429 rate-limit responses with exponential backoff (15s, 30s, 60s).
    Sleeps 2s between batches to stay under the 250k tokens/minute free-tier limit.

    Returns a flat list of embedding vectors (one per input text).
    """
    all_vectors: list[list[float]] = []
    total_batches = (len(texts) + EMBEDDING_BATCH - 1) // EMBEDDING_BATCH

    for batch_num, start in enumerate(
        tqdm(range(0, len(texts), EMBEDDING_BATCH), desc="Embedding", unit="batch"),
        start=1,
    ):
        batch = texts[start : start + EMBEDDING_BATCH]

        for attempt, wait in enumerate([0, 15, 30, 60], start=1):
            if wait:
                print(f"\n  Rate limited — waiting {wait}s before retry (attempt {attempt}/4) ...")
                time.sleep(wait)
            try:
                response = pc.inference.embed(
                    model      = PINECONE_EMBED_MODEL,
                    inputs     = batch,
                    parameters = {"input_type": input_type},
                )
                all_vectors.extend([item.values for item in response])
                if batch_num % 5 == 0 or batch_num == total_batches:
                    print(f"\n  Embedded {min(start + EMBEDDING_BATCH, len(texts))}/{len(texts)} chunks")
                break
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    if attempt == 4:
                        raise
                    continue
                raise

        # Stay under 250k tokens/minute free-tier cap (~2s between batches)
        time.sleep(2)

    return all_vectors


def load_chunks_json(path: Path = CHUNKS_JSON) -> list[dict]:
    """Load chunks from data/processed/chunks.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def upload_to_pinecone(
    index,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """
    Upload embeddings to Pinecone in batches of BATCH_SIZE.

    Vector ID format: "{source_document}_{chunk_index}_{sub_chunk_index}"
    sub_chunk_index is "0" when not present in metadata.

    Each batch is retried once after a 5-second wait on failure.
    Returns the total count of successfully uploaded vectors.
    """
    total    = len(chunks)
    uploaded = 0

    for batch_num, start in enumerate(range(0, total, BATCH_SIZE), start=1):
        batch_chunks = chunks[start : start + BATCH_SIZE]
        batch_embeds = embeddings[start : start + BATCH_SIZE]

        vectors = []
        for chunk, vec in zip(batch_chunks, batch_embeds):
            meta = chunk["metadata"]
            sub  = meta.get("sub_chunk_index")
            vid  = (
                f"{meta['source_document']}_"
                f"{meta['chunk_index']}_"
                f"{sub if sub is not None else '0'}"
            )
            vectors.append({
                "id":     vid,
                "values": vec,
                "metadata": {
                    "source_document": meta["source_document"],
                    "page_number":     meta.get("page_number") or 0,
                    "section_number":  meta.get("section_number") or "",
                    "chunk_index":     meta["chunk_index"],
                    "sub_chunk_index": sub if sub is not None else 0,
                    "char_count":      meta.get("char_count") or 0,
                },
            })

        for attempt in (1, 2):
            try:
                index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
                uploaded += len(vectors)
                print(f"  Uploaded batch {batch_num} — {uploaded}/{total} vectors")
                break
            except Exception as exc:
                if attempt == 1:
                    print(f"  Batch {batch_num} failed ({exc}), retrying in 5 s ...")
                    time.sleep(5)
                else:
                    print(
                        f"  Batch {batch_num} failed after retry — "
                        f"skipping {len(vectors)} vectors."
                    )

    return uploaded


def build_bm25_index(chunks: list[dict]) -> BM25Index:
    """
    Build a BM25Okapi index from chunk texts, wrap it in a runtime-safe
    BM25Index (retrieval.bm25_retriever.BM25Index), and serialise to
    retrieval/bm25_index.pkl.

    Immediately reloads the pickle and runs a smoke test: queries
    "licensing objectives" and confirms the top result is from
    section_182_guidance.pdf.  Logs the .pkl size in KB.
    """
    # Build ChunkView objects — the runtime-safe container pickled into .pkl
    chunk_views: list[ChunkView] = []
    for c in chunks:
        meta = c["metadata"]
        sub  = meta.get("sub_chunk_index")
        chunk_views.append(ChunkView(
            chunk_id        = (
                f"{meta['source_document']}_"
                f"{meta['chunk_index']}_"
                f"{sub if sub is not None else '0'}"
            ),
            source_document = meta["source_document"],
            text            = c["text"],
            bm25_text       = c["text"],
            heading         = meta.get("section_number") or "",
            section_number  = meta.get("section_number"),
            page_number     = meta.get("page_number"),
            chunk_index     = meta["chunk_index"],
            char_count      = meta.get("char_count", 0),
            sub_chunk_index = sub,
        ))

    # Tokenise
    print(f"  Tokenising {len(chunk_views)} chunks ...")
    corpus = [
        _tokenise(cv.bm25_text)
        for cv in tqdm(chunk_views, desc="Tokenising")
    ]

    # Build BM25Okapi and wrap in runtime-safe BM25Index
    bm25_model = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
    bm25_index = BM25Index(chunks=chunk_views, bm25_model=bm25_model)

    # Serialise — class is retrieval.bm25_retriever.BM25Index (runtime safe)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_index, f)

    size_kb = BM25_INDEX_PATH.stat().st_size / 1024
    print(f"  BM25 index saved: {BM25_INDEX_PATH} ({size_kb:.1f} KB)")

    # Smoke test — reload from disk, confirm pickle round-trip is clean
    print("  Smoke test: querying 'licensing objectives' ...")
    reloaded = BM25Index.load(BM25_INDEX_PATH)
    results  = reloaded.search("licensing objectives", top_k=1)
    if results:
        top = reloaded.get_chunk(results[0][0])
        print(f"  Top result: {top.source_document} (score={results[0][1]:.4f})")
        if top.source_document == "section_182_guidance.pdf":
            print("  [PASS] Top result is from section_182_guidance.pdf")
        else:
            print(
                f"  [WARN] Expected section_182_guidance.pdf, "
                f"got {top.source_document}"
            )
    else:
        print("  [WARN] Smoke test returned no results — check tokenisation.")

    return bm25_index


def main() -> None:
    t0 = time.time()

    print("=" * 60)
    print("LicenceBotUK Ingestion Pipeline")
    print("=" * 60)

    # 1. Load chunks
    print("\n[1/3] Loading chunks ...")
    chunks = load_chunks_json()
    print(f"  Total chunks loaded: {len(chunks)}")

    # 2. Embed via Pinecone Inference API (input_type="passage")
    print("\n[2/3] Embedding and uploading to Pinecone ...")
    if not PINECONE_API_KEY:
        raise EnvironmentError(
            "PINECONE_API_KEY is not set. Add it to .env before running ingestion."
        )

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = setup_pinecone_index(pc)

    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(pc, texts, input_type="passage")
    print(f"  Embedded {len(embeddings)} chunks.")

    # 4. Upload all vectors to Pinecone
    uploaded = upload_to_pinecone(index, chunks, embeddings)

    # 5. Verify Pinecone vector count matches chunks loaded
    time.sleep(5)  # allow Pinecone stats to settle after upsert
    stats          = index.describe_index_stats()
    pinecone_count = stats.get("total_vector_count", 0)
    if pinecone_count >= len(chunks):
        print(
            f"  [PASS] Pinecone vector count: {pinecone_count} "
            f"(expected >= {len(chunks)})"
        )
    else:
        print(
            f"  [WARN] Pinecone vector count {pinecone_count} "
            f"< chunks loaded {len(chunks)}"
        )

    # 6. Build and save BM25 index
    print("\n[3/3] Building BM25 index ...")
    build_bm25_index(chunks)

    # 7. Final summary
    elapsed = (time.time() - t0) / 60
    print("\n" + "=" * 60)
    print("LicenceBotUK ingestion complete")
    print(f"Vectors in Pinecone: {pinecone_count}")
    print(f"BM25 index saved: retrieval/bm25_index.pkl")
    print(f"Total time: {elapsed:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
