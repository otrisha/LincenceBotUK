"""
tests/e2e_retrieval_test.py
─────────────────────────────
End-to-end retrieval pipeline test.
Runs 6 queries through BM25, Pinecone, RRF, and GPT-4o-mini.
Shows per-query diagnostics and a final health summary.

Run:
    python tests/e2e_retrieval_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.bm25_retriever import get_bm25_index
from retrieval.dense_retriever import dense_search
from retrieval.rrf_fusion import hybrid_retrieve
from generation.generator import generate

# ── Queries ──────────────────────────────────────────────────────────────────

QUERIES = [
    "What are the four licensing objectives?",
    "How many TENs can a personal licence holder apply for per year?",
    "What ID is acceptable under Challenge 25?",
    "What happens if a DPS leaves the business?",
    "What are the mandatory conditions on every alcohol licence?",
    "What is the maximum capacity for a temporary event notice?",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def bar(char="=", width=60):
    return char * width

def section(title):
    print(f"\n{bar('=')}")
    print(f"  {title}")
    print(bar('='))

def subsection(title):
    print(f"\n  {bar('-', 56)}")
    print(f"  {title}")
    print(f"  {bar('-', 56)}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(bar('═'))
    print("  UK Hospitality RAG — End-to-End Retrieval Test")
    print(bar('═'))

    print("\nLoading BM25 index from disk ...")
    bm25 = get_bm25_index()
    print(f"  BM25 index loaded: {len(bm25)} chunks")

    # Tracking for summary
    all_docs_retrieved: dict[str, int] = {}
    weak_queries: list[str] = []
    uncited_queries: list[str] = []
    fallback_queries: list[str] = []

    for q_num, query in enumerate(QUERIES, start=1):
        section(f"Query {q_num}/6: {query}")

        # ── BM25 ──────────────────────────────────────────────────────────────
        subsection("BM25 Results (top 3)")
        bm25_raw = bm25.search(query, top_k=21)   # 3× over-fetch
        bm25_top3 = bm25_raw[:3]

        if not bm25_top3:
            print("  [WARN] BM25 returned no results.")
        else:
            for rank, (idx, score) in enumerate(bm25_top3, start=1):
                chunk = bm25.get_chunk(idx)
                section_ref = f" § {chunk.section_number}" if chunk.section_number else ""
                print(f"  #{rank}  score={score:.4f}  {chunk.source_document}{section_ref}")
                print(f"       heading: {chunk.heading[:70]}")

        # ── Dense (Pinecone) ──────────────────────────────────────────────────
        subsection("Pinecone Dense Results (top 3)")
        try:
            dense_top3 = dense_search(query, top_k=3)
            if not dense_top3:
                print("  [WARN] Dense search returned no results.")
            else:
                for rank, r in enumerate(dense_top3, start=1):
                    src = r.metadata.get("source_document", "unknown")
                    sec = r.metadata.get("section_number", "")
                    section_ref = f" § {sec}" if sec else ""
                    print(f"  #{rank}  score={r.score:.4f}  {src}{section_ref}")
        except Exception as exc:
            print(f"  [ERROR] Dense search failed: {exc}")
            dense_top3 = []

        # ── RRF Fusion ────────────────────────────────────────────────────────
        subsection("RRF Fused Results (top 3)")
        try:
            retrieved = hybrid_retrieve(query, bm25_index=bm25, top_k_final=5)
            rrf_top3 = retrieved[:3]

            if not rrf_top3:
                print("  [WARN] RRF returned no results — retrieval may be broken.")
                weak_queries.append(query)
            else:
                for rank, chunk in enumerate(rrf_top3, start=1):
                    section_ref = f" § {chunk.section_number}" if chunk.section_number else ""
                    sources = []
                    if chunk.in_bm25:  sources.append("BM25")
                    if chunk.in_dense: sources.append("Dense")
                    print(
                        f"  #{rank}  rrf={chunk.rrf_score:.5f}  "
                        f"{chunk.source_document}{section_ref}"
                        f"  [{'+'.join(sources)}]"
                    )
                    print(f"       heading: {chunk.heading[:70]}")

                # Track docs for summary
                for chunk in retrieved:
                    all_docs_retrieved[chunk.source_document] = (
                        all_docs_retrieved.get(chunk.source_document, 0) + 1
                    )

                if len(retrieved) < 3:
                    weak_queries.append(query)

        except Exception as exc:
            print(f"  [ERROR] RRF fusion failed: {exc}")
            retrieved = []
            weak_queries.append(query)

        # ── Generation ────────────────────────────────────────────────────────
        subsection("GPT-4o-mini Answer")
        try:
            response = generate(query, retrieved)
            print(f"\n  {response.answer[:1200]}")
            if len(response.answer) > 1200:
                print("  ... [truncated]")

            print(f"\n  Latency   : {response.latency_seconds:.2f}s")
            print(f"  Fallback  : {response.is_fallback}")
            print(f"  Safety    : {response.is_safety}")

            if response.citations:
                print(f"  Citations : {', '.join(response.citations[:8])}")
            else:
                print("  Citations : [NONE FOUND]")
                if not response.is_fallback:
                    uncited_queries.append(query)

            if response.is_fallback:
                fallback_queries.append(query)

        except Exception as exc:
            print(f"  [ERROR] Generation failed: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("RETRIEVAL HEALTH SUMMARY")

    print("\n  Documents retrieved (total chunk appearances across all queries):")
    for doc, count in sorted(all_docs_retrieved.items(), key=lambda x: -x[1]):
        bar_str = "#" * min(count, 40)
        print(f"    {doc:<40} {bar_str} ({count})")

    print("\n  Weak retrieval queries (< 3 results or RRF error):")
    if weak_queries:
        for q in weak_queries:
            print(f"    [!]{q}")
    else:
        print("    None — all queries returned ≥ 3 results [OK]")

    print("\n  Uncited answers (model answered without citing sources):")
    if uncited_queries:
        for q in uncited_queries:
            print(f"    [!]{q}")
    else:
        print("    None — all non-fallback answers included citations [OK]")

    print("\n  Fallback responses (model said 'insufficient information'):")
    if fallback_queries:
        for q in fallback_queries:
            print(f"    [!]{q}")
    else:
        print("    None [OK]")

    # Overall verdict
    issues = len(weak_queries) + len(uncited_queries) + len(fallback_queries)
    print(f"\n  Overall health: ", end="")
    if issues == 0:
        print("GOOD — no issues detected across all 6 queries [OK]")
    elif issues <= 2:
        print(f"ACCEPTABLE — {issues} minor issue(s) — see above")
    else:
        print(f"NEEDS ATTENTION — {issues} issue(s) — diagnose before deploying")

    print(f"\n{bar('═')}\n")


if __name__ == "__main__":
    run()
