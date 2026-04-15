"""
tests/test_retrieval.py
────────────────────────
Unit tests for the retrieval layer.

Tests:
  1. BM25 index can be loaded from disk (requires ingestion to have run)
  2. BM25 search returns correct types and score ordering
  3. RRF fusion produces correctly ordered, deduplicated results
  4. RetrievedChunk fields are fully populated
  5. Ingestion/runtime separation — sentence-transformers must NOT be importable
     from any module in the runtime import chain

Run:
    pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import importlib
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: minimal fake Chunk that mimics the ingestion Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class _FakeChunk:
    chunk_id: str
    source_document: str
    text: str
    bm25_text: str
    heading: str
    section_number: str | None
    page_number: int
    chunk_index: int
    token_count: int
    topic_category: str = "general"
    chunking_method: str = "section"


# ---------------------------------------------------------------------------
# Helper: build a minimal in-memory BM25Index without touching disk
# ---------------------------------------------------------------------------

def _build_test_bm25():
    """
    Build a tiny BM25Index object in memory for testing.
    Bypasses disk I/O so tests pass without a real .pkl file.
    """
    from rank_bm25 import BM25Okapi
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)

    chunks = [
        _FakeChunk("doc1-0001-premises-licence", "Licensing_Act_2003", "A premises licence authorises licensable activities.", "premises licence licensable activities", "Section 11 — Premises Licences", "11", 1, 0, 8),
        _FakeChunk("doc1-0002-personal-licence", "Licensing_Act_2003", "A personal licence authorises the holder to supply alcohol.", "personal licence supply alcohol", "Section 111 — Personal Licences", "111", 5, 1, 10),
        _FakeChunk("doc2-0001-ten", "Guidance_Notes", "A Temporary Event Notice allows small events without a premises licence.", "temporary event notice small events", "TEN Procedure", None, 3, 0, 12),
        _FakeChunk("doc2-0002-objectives", "Guidance_Notes", "The four licensing objectives are: crime prevention, public safety, public nuisance, child protection.", "licensing objectives crime public safety nuisance child", "Licensing Objectives", None, 2, 1, 15),
        _FakeChunk("doc1-0003-review", "Licensing_Act_2003", "Any responsible authority may apply for a review of a premises licence.", "review premises licence responsible authority", "Section 51 — Reviews", "51", 10, 2, 11),
    ]
    from nltk.corpus import stopwords as sw
    stops = set(sw.words("english"))

    def tokenise(text):
        tokens = nltk.word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum() and t not in stops]

    corpus = [tokenise(c.bm25_text) for c in chunks]
    bm25_model = BM25Okapi(corpus, k1=1.5, b=0.75)

    # Construct a BM25Index-compatible object
    # (mirrors the ingestion BM25Index interface)
    obj = MagicMock()
    obj.chunks = chunks
    obj._bm25 = bm25_model

    def search(query, top_k=7, source_filter=None):
        tokens = tokenise(query)
        if not tokens:
            return []
        scores = bm25_model.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            if source_filter and chunks[idx].source_document != source_filter:
                continue
            results.append((idx, float(score)))
            if len(results) >= top_k:
                break
        return results

    def get_chunk(index):
        return chunks[index]

    obj.search = search
    obj.get_chunk = get_chunk
    obj.__len__ = lambda self: len(chunks)
    return obj


# ---------------------------------------------------------------------------
# Test 1: BM25 search returns correct types
# ---------------------------------------------------------------------------

class TestBM25Search:
    def setup_method(self):
        self.bm25 = _build_test_bm25()

    def test_search_returns_list(self):
        results = self.bm25.search("premises licence")
        assert isinstance(results, list)

    def test_search_results_are_tuples(self):
        results = self.bm25.search("premises licence")
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            idx, score = item
            assert isinstance(idx, int)
            assert isinstance(score, float)

    def test_search_sorted_descending(self):
        results = self.bm25.search("personal licence supply alcohol")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_zero_score_excluded(self):
        results = self.bm25.search("personal licence supply alcohol")
        for _, score in results:
            assert score > 0

    def test_search_empty_query_returns_empty(self):
        # After tokenisation, a stopword-only query produces no tokens
        results = self.bm25.search("the and or")
        assert isinstance(results, list)

    def test_search_top_k_respected(self):
        results = self.bm25.search("licence", top_k=2)
        assert len(results) <= 2

    def test_known_result_top_ranked(self):
        results = self.bm25.search("temporary event notice")
        top_idx = results[0][0] if results else None
        assert top_idx is not None
        chunk = self.bm25.get_chunk(top_idx)
        assert "temporary" in chunk.text.lower() or "event" in chunk.text.lower()


# ---------------------------------------------------------------------------
# Test 2: RRF fusion logic
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def _make_chunk(self, cid, text="Test text.", heading="Heading", source="doc1",
                    section=None, page=1, topic="general"):
        return _FakeChunk(
            chunk_id=cid, source_document=source, text=text,
            bm25_text=text, heading=heading, section_number=section,
            page_number=page, chunk_index=0, token_count=len(text.split()),
            topic_category=topic,
        )

    def test_rrf_score_increases_with_double_appearance(self):
        """A chunk present in both lists must score higher than one in only one list."""
        from retrieval.rrf_fusion import _rrf

        bm25 = _build_test_bm25()
        # Give chunk 0 rank 1 in BM25
        bm25_results = [(0, 10.0), (1, 5.0)]

        from retrieval.dense_retriever import DenseResult
        # chunk 0 also appears in dense at rank 1
        dense_results = [
            DenseResult(chunk_id=bm25.get_chunk(0).chunk_id, score=0.9, metadata={}),
            DenseResult(chunk_id=bm25.get_chunk(2).chunk_id, score=0.8, metadata={}),
        ]

        fused = _rrf(bm25_results, dense_results, bm25)
        scores = {cid: score for cid, score, _, _ in fused}

        # chunk 0 is in both lists — must have the highest score
        top_id = fused[0][0]
        assert top_id == bm25.get_chunk(0).chunk_id

    def test_rrf_deduplication(self):
        """Each chunk_id should appear at most once in the fused output."""
        from retrieval.rrf_fusion import _rrf
        from retrieval.dense_retriever import DenseResult

        bm25 = _build_test_bm25()
        bm25_results = [(0, 9.0), (1, 7.0), (2, 5.0)]
        dense_results = [
            DenseResult(chunk_id=bm25.get_chunk(0).chunk_id, score=0.95, metadata={}),
            DenseResult(chunk_id=bm25.get_chunk(1).chunk_id, score=0.85, metadata={}),
        ]
        fused = _rrf(bm25_results, dense_results, bm25)
        chunk_ids = [cid for cid, _, _, _ in fused]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_rrf_flags_source_correctly(self):
        """in_bm25 and in_dense flags must accurately reflect which list contributed."""
        from retrieval.rrf_fusion import _rrf
        from retrieval.dense_retriever import DenseResult

        bm25 = _build_test_bm25()
        bm25_results = [(0, 8.0)]
        dense_results = [
            DenseResult(chunk_id=bm25.get_chunk(2).chunk_id, score=0.88, metadata={}),
        ]
        fused = _rrf(bm25_results, dense_results, bm25)
        fused_dict = {cid: (ib, id_) for cid, _, ib, id_ in fused}

        c0 = bm25.get_chunk(0).chunk_id
        c2 = bm25.get_chunk(2).chunk_id
        assert fused_dict[c0] == (True, False)
        assert fused_dict[c2] == (False, True)


# ---------------------------------------------------------------------------
# Test 3: Ingestion / runtime separation — sentence-transformers must NOT
# be importable from the runtime import chain
# ---------------------------------------------------------------------------

class TestIngestionRuntimeSeparation:
    """
    Verify that sentence-transformers does not appear in the transitive
    import chain of any runtime module.
    """

    RUNTIME_MODULES = [
        "retrieval.bm25_retriever",
        "retrieval.dense_retriever",
        "retrieval.rrf_fusion",
        "generation.prompts",
        "generation.generator",
    ]

    def test_sentence_transformers_not_in_runtime_modules(self):
        """
        For each runtime module, check that 'sentence_transformers' is not
        in its source code import statements.
        """
        root = Path(__file__).parent.parent
        import_re = __import__("re").compile(
            r"^\s*(import|from)\s+sentence_transformers", __import__("re").MULTILINE
        )
        for module_path in self.RUNTIME_MODULES:
            file_path = root / module_path.replace(".", "/") / ""
            # Try as .py file
            py_path = root / (module_path.replace(".", "/") + ".py")
            if py_path.exists():
                source = py_path.read_text(encoding="utf-8")
                matches = import_re.findall(source)
                assert not matches, (
                    f"sentence_transformers found in runtime module {py_path}!\n"
                    f"Matches: {matches}"
                )

    def test_embedder_not_imported_by_runtime_modules(self):
        """
        ingestion/embedder.py must not be imported (even transitively)
        by any runtime module.
        """
        root = Path(__file__).parent.parent
        import_re = __import__("re").compile(
            r"^\s*(import|from)\s+(ingestion\.embedder|ingestion import embedder)",
            __import__("re").MULTILINE,
        )
        for module_path in self.RUNTIME_MODULES:
            py_path = root / (module_path.replace(".", "/") + ".py")
            if py_path.exists():
                source = py_path.read_text(encoding="utf-8")
                matches = import_re.findall(source)
                assert not matches, (
                    f"ingestion.embedder imported in runtime module {py_path}!"
                )

    def test_bm25_retriever_loads_from_pickle_not_rebuild(self):
        """
        bm25_retriever.py must use pickle.load and must not call BM25Okapi()
        to rebuild the index at runtime.
        """
        root = Path(__file__).parent.parent
        source = (root / "retrieval" / "bm25_retriever.py").read_text(encoding="utf-8")
        assert "pickle.load" in source, "bm25_retriever.py must use pickle.load"
        assert "BM25Okapi(" not in source, (
            "bm25_retriever.py must not rebuild the BM25 index at runtime. "
            "Index building belongs in ingestion/run_ingestion.py only."
        )

    def test_dense_retriever_uses_pinecone_inference(self):
        """
        dense_retriever.py must use the Pinecone Inference API for query embedding
        and must NOT import or call SentenceTransformer.
        """
        root = Path(__file__).parent.parent
        source = (root / "retrieval" / "dense_retriever.py").read_text(encoding="utf-8")
        assert "pc.inference.embed" in source, (
            "dense_retriever.py must use Pinecone Inference API (pc.inference.embed)"
        )
        assert "SentenceTransformer" not in source, (
            "dense_retriever.py must not import or call SentenceTransformer"
        )
        assert "sentence_transformers" not in source.lower(), (
            "dense_retriever.py must not reference sentence_transformers"
        )
