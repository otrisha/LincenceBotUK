"""
retrieval/bm25_retriever.py
────────────────────────────
Loads the pre-built BM25 index from retrieval/bm25_index.pkl and
exposes a search interface used by rrf_fusion.py.

Runtime contract:
  - The .pkl file is written by ingestion/run_ingestion.py (offline step).
  - This module uses pickle.load() — it never rebuilds the index.
  - sentence-transformers is NOT imported here or anywhere in its import chain.
  - The BM25Index class defined in this file is a read-only runtime copy;
    the full class (with save/build) lives in ingestion/run_ingestion.py.

SAFE for runtime import: no heavy dependencies, no local model.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nltk
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
BM25_INDEX_PATH = Path(
    os.getenv("BM25_INDEX_PATH", ROOT / "retrieval" / "bm25_index.pkl")
)
CHUNKS_JSON_PATH = ROOT / "data" / "processed" / "chunks.json"

BM25_K1: float = float(os.getenv("BM25_K1", 1.5))
BM25_B:  float = float(os.getenv("BM25_B",  0.75))

TOP_K_BM25: int = int(os.getenv("TOP_K_BM25", 7))

# ---------------------------------------------------------------------------
# Ensure NLTK resources at import time (lightweight, cached after first run)
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk()

from nltk.corpus import stopwords as _sw_corpus
_STOPWORDS: set[str] = set(_sw_corpus.words("english"))


def _tokenise(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Runtime-safe Chunk stub
# The full Chunk dataclass lives in ingestion/chunker.py (ingestion only).
# At runtime we only need a minimal read view — deserialized from the .pkl.
# ---------------------------------------------------------------------------

@dataclass
class ChunkView:
    """
    Minimal runtime view of a chunk stored inside the BM25 index pickle.

    Required fields (no default) must come first; optional fields at the end.
    rrf_fusion.py accesses: chunk_id, source_document, text, heading,
    section_number, page_number, topic_category.
    """
    chunk_id:        str
    source_document: str
    text:            str
    bm25_text:       str
    heading:         str
    # --- optional / defaulted fields ---
    section_number:  Optional[str] = None
    page_number:     Optional[int] = None   # None for TXT-sourced chunks
    chunk_index:     int           = 0
    token_count:     int           = 0
    char_count:      int           = 0
    sub_chunk_index: Optional[int] = None
    topic_category:  str           = "general"
    chunking_method: str           = "section"


# ---------------------------------------------------------------------------
# BM25Index — runtime-safe loader
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Read-only runtime wrapper around the pickled BM25 index.

    The pickle was produced by ingestion/run_ingestion.BM25Index.
    Python and rank-bm25 versions must match between ingestion and runtime.
    """

    def __init__(self, chunks: list, bm25_model: BM25Okapi) -> None:
        self.chunks = chunks
        self._bm25 = bm25_model

    @classmethod
    def load(cls, path: Path = BM25_INDEX_PATH) -> "BM25Index":
        """
        Deserialise the BM25 index from disk.

        Falls back to rebuilding from chunks.json if the pkl is missing
        (e.g. on Streamlit Cloud where ingestion cannot run).
        """
        if not path.exists():
            if CHUNKS_JSON_PATH.exists():
                return cls._build_from_chunks_json()
            raise FileNotFoundError(
                f"BM25 index not found at {path} and no chunks.json at "
                f"{CHUNKS_JSON_PATH}. Run ingestion first."
            )
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    @classmethod
    def _build_from_chunks_json(cls, path: Path = CHUNKS_JSON_PATH) -> "BM25Index":
        """
        Build the BM25 index in-memory from data/processed/chunks.json.

        Used as a fallback on Streamlit Cloud where the pre-built pkl is not
        available. Takes ~1s for 931 chunks. The result is NOT saved to disk
        (Streamlit Cloud's filesystem is ephemeral and read-only in /mount/src).
        """
        import json
        print(f"BM25 pkl not found — rebuilding from {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunk_views: list[ChunkView] = []
        for c in data:
            meta = c.get("metadata", c)
            sub = meta.get("sub_chunk_index")
            chunk_id = (
                f"{meta['source_document']}_"
                f"{meta['chunk_index']}_"
                f"{sub if sub is not None else '0'}"
            )
            chunk_views.append(ChunkView(
                chunk_id        = chunk_id,
                source_document = meta.get("source_document", ""),
                text            = c.get("text", ""),
                bm25_text       = c.get("text", ""),
                heading         = meta.get("section_number") or "",
                section_number  = meta.get("section_number"),
                page_number     = meta.get("page_number"),
                chunk_index     = meta.get("chunk_index", 0),
                char_count      = meta.get("char_count", 0),
                sub_chunk_index = sub,
            ))

        corpus = [_tokenise(cv.bm25_text) for cv in chunk_views]
        bm25_model = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
        print(f"BM25 index rebuilt from {len(chunk_views)} chunks.")
        return cls(chunks=chunk_views, bm25_model=bm25_model)

    def search(
        self,
        query: str,
        top_k: int = TOP_K_BM25,
        source_filter: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """
        Score all chunks against the query tokens.

        Args:
            query:         Raw query string (will be tokenised internally).
            top_k:         Number of results to return.
            source_filter: If set, only return chunks from this source_document.

        Returns:
            List of (chunk_index, score) sorted by score descending.
            Zero-score chunks are excluded.
        """
        tokens = _tokenise(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            if source_filter and self.chunks[idx].source_document != source_filter:
                continue
            results.append((idx, float(score)))
            if len(results) >= top_k:
                break
        return results

    def get_chunk(self, index: int):
        return self.chunks[index]

    def __len__(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Cached singleton — loaded once per Streamlit worker via st.cache_resource
# ---------------------------------------------------------------------------

_bm25_singleton: Optional[BM25Index] = None


def get_bm25_index(path: Path = BM25_INDEX_PATH) -> BM25Index:
    """
    Return the cached BM25 index, loading from disk on first call.

    In Streamlit, wrap this with @st.cache_resource in streamlit_app.py
    to ensure it is loaded once per worker and not per user request.
    """
    global _bm25_singleton
    if _bm25_singleton is None:
        _bm25_singleton = BM25Index.load(path)
    return _bm25_singleton
