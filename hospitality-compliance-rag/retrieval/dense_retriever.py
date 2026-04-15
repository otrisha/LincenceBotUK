"""
retrieval/dense_retriever.py
──────────────────────────────
Dense retrieval using Pinecone's hosted inference API for query embedding.

⚠️  NO local embedding model is loaded here. ⚠️
Query embedding at runtime is delegated entirely to the Pinecone Inference API.
This keeps sentence-transformers and torch out of the runtime dependency tree.

The Pinecone-hosted model (PINECONE_EMBED_MODEL) must match the model used
during ingestion (ingestion/embedder.py EMBEDDING_MODEL). Both must produce
vectors of dimension PINECONE_DIMENSION.

SAFE for runtime import: depends only on pinecone, numpy, python-dotenv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from pinecone import Pinecone

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hospitality-compliance")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "hospitality-kb")
PINECONE_EMBED_MODEL= os.getenv("PINECONE_EMBED_MODEL", "multilingual-e5-large")

TOP_K_DENSE: int = int(os.getenv("TOP_K_DENSE", 7))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DenseResult:
    chunk_id: str
    score: float
    metadata: dict


# ---------------------------------------------------------------------------
# Pinecone client — lazy singleton
# ---------------------------------------------------------------------------

_pc_client: Optional[Pinecone] = None
_pc_index = None


def _get_index():
    global _pc_client, _pc_index
    if _pc_index is None:
        if not PINECONE_API_KEY:
            raise EnvironmentError(
                "PINECONE_API_KEY is not set. "
                "Add it to .env or Streamlit secrets."
            )
        _pc_client = Pinecone(api_key=PINECONE_API_KEY)
        _pc_index = _pc_client.Index(PINECONE_INDEX_NAME)
    return _pc_index


def get_pinecone_client() -> Pinecone:
    """Return (or create) the cached Pinecone client."""
    global _pc_client
    if _pc_client is None:
        _pc_client = Pinecone(api_key=PINECONE_API_KEY)
    return _pc_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dense_search(
    query: str,
    top_k: int = TOP_K_DENSE,
    metadata_filter: Optional[dict] = None,
) -> list[DenseResult]:
    """
    Embed query via Pinecone Inference API and search the dense index.

    The Pinecone Inference API handles embedding server-side — no local model
    is loaded or called in this function.

    Args:
        query:           Raw user query string.
        top_k:           Number of candidates to retrieve.
        metadata_filter: Optional Pinecone filter dict, e.g.
                         {"topic_category": {"$eq": "personal_licence"}}

    Returns:
        List of DenseResult sorted by score descending.
    """
    pc = get_pinecone_client()

    # --- Embed via Pinecone Inference (no local model) ---------------------
    embed_response = pc.inference.embed(
        model=PINECONE_EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query"},
    )
    query_vector = embed_response[0].values

    # --- Query the index ---------------------------------------------------
    index = _get_index()
    query_kwargs: dict = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": PINECONE_NAMESPACE,
    }
    if metadata_filter:
        query_kwargs["filter"] = metadata_filter

    response = index.query(**query_kwargs)

    results = [
        DenseResult(
            chunk_id=match["id"],
            score=float(match["score"]),
            metadata=match.get("metadata", {}),
        )
        for match in response.get("matches", [])
    ]
    return results
