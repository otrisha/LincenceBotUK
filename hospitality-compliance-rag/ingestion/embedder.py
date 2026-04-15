"""
ingestion/embedder.py
──────────────────────
Loads sentence-transformers and encodes all chunks for Pinecone upsert.

⚠️  INGESTION ONLY ⚠️
This module imports sentence-transformers and torch.
It MUST NOT be imported — directly or transitively — by:
  - app/streamlit_app.py
  - retrieval/bm25_retriever.py
  - retrieval/dense_retriever.py
  - retrieval/rrf_fusion.py
  - generation/generator.py

At runtime, query embedding is handled by the Pinecone Inference API
inside retrieval/dense_retriever.py — no local model is loaded.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

# sentence-transformers is intentionally imported here and nowhere else.
from sentence_transformers import SentenceTransformer

from ingestion.chunker import Chunk

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Must match PINECONE_EMBED_MODEL and PINECONE_DIMENSION in .env.example.
# multilingual-e5-large outputs 1024 dims (matches .env.example default).
# If you switch to all-MiniLM-L6-v2 (384 dims), update PINECONE_DIMENSION too.
EMBEDDING_MODEL: str = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "intfloat/multilingual-e5-large",
)
EMBEDDING_BATCH: int = int(os.getenv("EMBEDDING_BATCH", 32))


# ---------------------------------------------------------------------------
# Singleton — model loaded once per ingestion run, never at runtime
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Singleton wrapper around SentenceTransformer.

    Loaded once per ingestion process. The singleton is scoped to the
    ingestion script process; it is never accessible at Streamlit runtime.
    """

    _instance: SentenceTransformer | None = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._instance is None:
            print(f"  Loading embedding model: {EMBEDDING_MODEL} ...")
            cls._instance = SentenceTransformer(EMBEDDING_MODEL)
            print(f"  Model loaded. Output dimension: {cls._instance.get_sentence_embedding_dimension()}")
        return cls._instance

    @classmethod
    def encode_chunks(cls, chunks: list[Chunk]) -> np.ndarray:
        """
        Encode all chunk texts in batches.

        Returns float32 ndarray of shape (n_chunks, embedding_dim).
        Embeddings are L2-normalised (required for cosine metric in Pinecone).
        """
        model = cls.get()
        texts = [c.text for c in chunks]
        print(f"  Encoding {len(texts)} chunks in batches of {EMBEDDING_BATCH} ...")
        vectors = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH,
            normalize_embeddings=True,   # cosine ≡ dot-product on unit vectors
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)
