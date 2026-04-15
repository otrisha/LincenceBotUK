"""
evaluation/ragas_eval.py
─────────────────────────
RAGAS-based evaluation pipeline for the hospitality compliance RAG system.

Metrics computed:
  - faithfulness          — claims in answer supported by retrieved context
  - answer_relevancy      — answer addresses the question
  - context_precision     — retrieved contexts ranked well for relevance
  - context_recall        — context covers the ground truth answer

Retrieval metrics (no RAGAS needed):
  - Recall@5    — is the relevant document present in the top-5 chunks?
  - Precision@5 — fraction of top-5 chunks from the relevant document
  - MRR         — Mean Reciprocal Rank of first relevant chunk

Output: timestamped CSV saved to evaluation/results/.

INGESTION / EVALUATION ONLY — do not run in the Streamlit app.
Requires requirements_ingestion.txt (ragas, langchain, langchain-openai, datasets).

Usage:
    python -m evaluation.ragas_eval --mode hybrid
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load .env before any module reads env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from retrieval.bm25_retriever import BM25Index, get_bm25_index
from retrieval.rrf_fusion import hybrid_retrieve, RetrievedChunk
from generation.generator import generate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
TEST_QUESTIONS_PATH = Path(__file__).parent / "test_questions.txt"

EVAL_TARGETS = {
    "recall_at_5":             0.80,
    "precision_at_5":          0.40,
    "mrr":                     0.70,
    "ragas_faithfulness":      0.80,
    "ragas_answer_relevancy":  0.75,
    "ragas_context_precision": 0.65,
    "ragas_context_recall":    0.75,
    "max_latency_seconds":     8.0,
}

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Question loader
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    query_id: str
    topic_category: str
    question: str
    ground_truth: str = ""        # populated externally for context_recall
    relevant_doc: str = ""        # source_document expected in top-5


def load_questions(path: Path = TEST_QUESTIONS_PATH) -> list[EvalQuestion]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            questions.append(EvalQuestion(
                query_id=parts[0],
                topic_category=parts[1],
                question=parts[2],
                ground_truth=parts[3] if len(parts) > 3 else "",
                relevant_doc=parts[4] if len(parts) > 4 else "",
            ))
    return questions


# ---------------------------------------------------------------------------
# Retrieval metrics (no RAGAS)
# ---------------------------------------------------------------------------

def _recall_at_k(retrieved: list[RetrievedChunk], relevant_doc: str, k: int = 5) -> float:
    if not relevant_doc:
        return float("nan")
    return float(
        any(c.source_document == relevant_doc for c in retrieved[:k])
    )


def _precision_at_k(retrieved: list[RetrievedChunk], relevant_doc: str, k: int = 5) -> float:
    if not relevant_doc or not retrieved:
        return float("nan")
    relevant_count = sum(1 for c in retrieved[:k] if c.source_document == relevant_doc)
    return relevant_count / min(k, len(retrieved))


def _mrr(retrieved: list[RetrievedChunk], relevant_doc: str) -> float:
    if not relevant_doc:
        return float("nan")
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.source_document == relevant_doc:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(mode: str = "hybrid") -> pd.DataFrame:
    """
    Run the full evaluation over all test questions.

    mode: "hybrid" | "bm25_only" | "dense_only"
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    print(f"\nLoaded {len(questions)} evaluation questions.")

    # Load BM25 index once
    bm25_index = get_bm25_index()

    records = []
    ragas_inputs = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    print(f"\nRunning evaluation (mode={mode}) ...")
    for q in questions:
        t0 = time.perf_counter()

        # Retrieve
        if mode == "bm25_only":
            bm25_raw = bm25_index.search(q.question, top_k=5)
            retrieved = [
                RetrievedChunk(
                    chunk_id=bm25_index.get_chunk(i).chunk_id,
                    source_document=bm25_index.get_chunk(i).source_document,
                    text=bm25_index.get_chunk(i).text,
                    heading=bm25_index.get_chunk(i).heading,
                    section_number=getattr(bm25_index.get_chunk(i), "section_number", None),
                    page_number=bm25_index.get_chunk(i).page_number,
                    topic_category=bm25_index.get_chunk(i).topic_category,
                    rrf_score=score,
                    in_bm25=True,
                    in_dense=False,
                )
                for i, score in bm25_raw[:5]
            ]
        elif mode == "dense_only":
            from retrieval.dense_retriever import dense_search
            dense_raw = dense_search(q.question, top_k=5)
            retrieved = []
            for dr in dense_raw:
                # resolve text from BM25 index
                for idx in range(len(bm25_index)):
                    c = bm25_index.get_chunk(idx)
                    if c.chunk_id == dr.chunk_id:
                        retrieved.append(RetrievedChunk(
                            chunk_id=c.chunk_id,
                            source_document=c.source_document,
                            text=c.text,
                            heading=c.heading,
                            section_number=getattr(c, "section_number", None),
                            page_number=c.page_number,
                            topic_category=c.topic_category,
                            rrf_score=dr.score,
                            in_bm25=False,
                            in_dense=True,
                        ))
                        break
        else:
            retrieved = hybrid_retrieve(q.question, bm25_index=bm25_index)

        # Generate
        response = generate(q.question, retrieved)
        latency = time.perf_counter() - t0

        # Retrieval metrics
        recall = _recall_at_k(retrieved, q.relevant_doc)
        precision = _precision_at_k(retrieved, q.relevant_doc)
        mrr = _mrr(retrieved, q.relevant_doc)

        records.append({
            "query_id":          q.query_id,
            "topic_category":    q.topic_category,
            "question":          q.question,
            "answer":            response.answer,
            "is_fallback":       response.is_fallback,
            "citations_count":   len(response.citations),
            "retrieved_count":   response.retrieved_count,
            "recall_at_5":       recall,
            "precision_at_5":    precision,
            "mrr":               mrr,
            "latency_seconds":   round(latency, 3),
        })

        ragas_inputs["question"].append(q.question)
        ragas_inputs["answer"].append(response.answer)
        ragas_inputs["contexts"].append([c.text for c in retrieved])
        ragas_inputs["ground_truth"].append(q.ground_truth or response.answer)

        print(f"  [{q.query_id}] recall={recall:.2f} mrr={mrr:.2f} "
              f"latency={latency:.2f}s")

    # --- RAGAS metrics -------------------------------------------------------
    print("\nRunning RAGAS evaluation ...")
    llm = ChatOpenAI(model=OPENAI_MODEL, openai_api_key=os.environ["OPENAI_API_KEY"])
    ragas_dataset = Dataset.from_dict(ragas_inputs)
    ragas_results = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
    )
    ragas_df = ragas_results.to_pandas()

    # --- Merge ---------------------------------------------------------------
    df = pd.DataFrame(records)
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in ragas_df.columns:
            df[f"ragas_{col}"] = ragas_df[col].values

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for metric, target in EVAL_TARGETS.items():
        col_map = {
            "recall_at_5":             "recall_at_5",
            "precision_at_5":          "precision_at_5",
            "mrr":                     "mrr",
            "ragas_faithfulness":      "ragas_faithfulness",
            "ragas_answer_relevancy":  "ragas_answer_relevancy",
            "ragas_context_precision": "ragas_context_precision",
            "ragas_context_recall":    "ragas_context_recall",
            "max_latency_seconds":     "latency_seconds",
        }
        col = col_map.get(metric)
        if col and col in df.columns:
            actual = df[col].mean()
            status = "PASS" if (
                actual >= target if metric != "max_latency_seconds" else actual <= target
            ) else "FAIL"
            print(f"  {metric:<30} {actual:.3f}  (target {'≥' if metric != 'max_latency_seconds' else '≤'}{target})  [{status}]")

    # --- Save CSV -----------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{mode}_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "bm25_only", "dense_only"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    args = parser.parse_args()
    run_evaluation(mode=args.mode)
