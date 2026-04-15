"""
retrieval/rrf_fusion.py
────────────────────────
Reciprocal Rank Fusion (RRF) over BM25 and dense retrieval results.

Formula (Robertson & Zaragoza, 2009):
    rrf_score(doc) = Σ  1 / (k + rank)
                    sources

where k = RRF_K = 60 (the standard smoothing constant).

A chunk appearing in both BM25 and dense results accumulates contributions
from both sources; chunks in only one list still receive a score.

SAFE for runtime import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from retrieval.bm25_retriever import BM25Index, get_bm25_index
from retrieval.dense_retriever import DenseResult, dense_search

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RRF_K: int       = int(os.getenv("RRF_K", 60))
TOP_K_BM25: int  = int(os.getenv("TOP_K_BM25", 7))
TOP_K_DENSE: int = int(os.getenv("TOP_K_DENSE", 7))
TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", 5))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A fused retrieval result ready for the prompt builder."""
    chunk_id: str
    source_document: str
    text: str
    heading: str
    section_number: Optional[str]
    page_number: int
    topic_category: str
    rrf_score: float
    in_bm25: bool
    in_dense: bool


# ---------------------------------------------------------------------------
# RRF core
# ---------------------------------------------------------------------------

def _rrf(
    bm25_results: list[tuple[int, float]],   # (chunk_index, bm25_score)
    dense_results: list[DenseResult],
    bm25_index: BM25Index,
    k: int = RRF_K,
) -> list[tuple[str, float, bool, bool]]:
    """
    Fuse BM25 and dense results using RRF.

    Returns list of (chunk_id, rrf_score, in_bm25, in_dense) sorted
    by rrf_score descending.
    """
    rrf_scores: dict[str, float]   = {}
    in_bm25:    dict[str, bool]    = {}
    in_dense:   dict[str, bool]    = {}

    # BM25 contribution
    for rank, (chunk_idx, _) in enumerate(bm25_results, start=1):
        chunk = bm25_index.get_chunk(chunk_idx)
        cid = chunk.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        in_bm25[cid] = True

    # Dense contribution
    for rank, result in enumerate(dense_results, start=1):
        cid = result.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        in_dense[cid] = True

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        (cid, score, in_bm25.get(cid, False), in_dense.get(cid, False))
        for cid, score in merged
    ]


# ---------------------------------------------------------------------------
# Chunk text resolution
# ---------------------------------------------------------------------------

def _resolve_chunk_text(
    chunk_id: str,
    bm25_index: BM25Index,
    dense_result: Optional[DenseResult],
) -> Optional[object]:
    """
    Look up the full chunk object from the BM25 index by chunk_id.

    Dense Pinecone results do NOT store text (by design — see CLAUDE.md
    Known Limitation #3 equivalent). The BM25 index is the authoritative
    text store at runtime.
    """
    # Build a lookup dict on first call — O(n) but done once per request
    # (optimise by caching at HybridRetriever level if needed)
    for i in range(len(bm25_index)):
        chunk = bm25_index.get_chunk(i)
        if chunk.chunk_id == chunk_id:
            return chunk
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    bm25_index: Optional[BM25Index] = None,
    metadata_filter: Optional[dict] = None,
    top_k_bm25: int = TOP_K_BM25,
    top_k_dense: int = TOP_K_DENSE,
    top_k_final: int = TOP_K_FINAL,
) -> list[RetrievedChunk]:
    """
    Run BM25 and dense retrieval in parallel, fuse with RRF, return top chunks.

    Args:
        query:           Raw user query (not pre-processed).
        bm25_index:      Pre-loaded BM25Index; if None, loaded from default path.
        metadata_filter: Optional Pinecone metadata filter for targeted retrieval.
        top_k_bm25:      BM25 candidates to collect before fusion.
        top_k_dense:     Dense candidates to collect before fusion.
        top_k_final:     Final number of chunks to pass to the generator.

    Returns:
        List of RetrievedChunk sorted by RRF score descending (best first).
    """
    if bm25_index is None:
        bm25_index = get_bm25_index()

    # Over-fetch BM25 by 3x to absorb filter losses
    bm25_raw = bm25_index.search(query, top_k=top_k_bm25 * 3)
    bm25_results = bm25_raw[:top_k_bm25]

    # Dense retrieval (embedding handled server-side by Pinecone)
    dense_results = dense_search(query, top_k=top_k_dense, metadata_filter=metadata_filter)

    # Fuse
    fused = _rrf(bm25_results, dense_results, bm25_index)

    # Build lookup of chunk_id → DenseResult for metadata enrichment
    dense_map = {r.chunk_id: r for r in dense_results}

    # Resolve chunk text and build final results
    final: list[RetrievedChunk] = []
    for chunk_id, rrf_score, in_bm25, in_dense in fused:
        chunk = _resolve_chunk_text(chunk_id, bm25_index, dense_map.get(chunk_id))
        if chunk is None:
            # Chunk present in Pinecone but not in BM25 index — index mismatch
            continue
        final.append(RetrievedChunk(
            chunk_id=chunk_id,
            source_document=chunk.source_document,
            text=chunk.text,
            heading=chunk.heading,
            section_number=getattr(chunk, "section_number", None),
            page_number=chunk.page_number,
            topic_category=chunk.topic_category,
            rrf_score=rrf_score,
            in_bm25=in_bm25,
            in_dense=in_dense,
        ))
        if len(final) >= top_k_final:
            break

    return final
