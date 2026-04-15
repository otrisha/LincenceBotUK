# CLAUDE.md — UK Hospitality Licensing Compliance Hybrid RAG
**LCR-RAG-001 | Hybrid RAG Chatbot for UK Hospitality Licensing**

> This file is the authoritative reference for the architecture, design decisions,
> ingestion/runtime separation rules, and production requirements of this project.
> Read it entirely before modifying any file.

---

## Architecture Overview

A **Hybrid Retrieval-Augmented Generation (RAG)** system that answers UK
hospitality licensing compliance questions using the Licensing Act 2003 and
associated guidance documents as the exclusive knowledge source.

```
INGESTION PHASE (offline, run once)
────────────────────────────────────────────────────────────────────────
 PDF Documents (data/raw/)
     │
     ▼
 ingestion/document_loader.py    ← pdfplumber extracts blocks with metadata
     │
     ▼
 ingestion/chunker.py            ← domain-adaptive chunking → Chunk objects
     │
     ├──▶ BM25Index              ← serialised to retrieval/bm25_index.pkl
     │
     └──▶ ingestion/embedder.py  ← sentence-transformers (INGESTION ONLY)
                                        │
                                        ▼
                                    Pinecone upsert (namespace: hospitality-kb)

RUNTIME PHASE (every user query via Streamlit)
────────────────────────────────────────────────────────────────────────
 User Query
     │
     ▼
 retrieval/bm25_retriever.py     ← loads .pkl, tokenises, BM25Okapi.get_scores()
     │
 retrieval/dense_retriever.py    ← Pinecone Inference API embeds query (NO local model)
     │
     ▼
 retrieval/rrf_fusion.py         ← RRF (k=60) fuses both lists → top 5 chunks
     │
     ▼
 generation/prompts.py           ← builds system prompt with context passages
     │
     ▼
 generation/generator.py         ← GPT-4o-mini, citation extraction, fallback detection
     │
     ▼
 app/streamlit_app.py            ← renders answer + source citations panel
```

---

## Critical Rule: Ingestion vs Runtime Separation

**sentence-transformers and torch must NEVER appear in the runtime import chain.**

| Module | Allowed at runtime? |
|--------|---------------------|
| `ingestion/embedder.py` | NO — ingestion only |
| `ingestion/chunker.py` | NO — ingestion only |
| `ingestion/document_loader.py` | NO — ingestion only |
| `ingestion/run_ingestion.py` | NO — entry point only |
| `retrieval/bm25_retriever.py` | YES |
| `retrieval/dense_retriever.py` | YES |
| `retrieval/rrf_fusion.py` | YES |
| `generation/prompts.py` | YES |
| `generation/generator.py` | YES |
| `app/streamlit_app.py` | YES |

**Enforcement:** `tests/test_retrieval.py::TestIngestionRuntimeSeparation`
verifies this rule automatically. Run before every deployment.

---

## Query Embedding at Runtime

At runtime, user queries are embedded by the **Pinecone Inference API**:

```python
embed_response = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={"input_type": "query"},
)
```

The `PINECONE_EMBED_MODEL` must match the model used in `ingestion/embedder.py`
during ingestion. Both must produce vectors of `PINECONE_DIMENSION` dimensions.

---

## File Structure

```
hospitality-compliance-rag/
├── CLAUDE.md                          ← This file
├── requirements_ingestion.txt         ← Heavy deps (sentence-transformers, ragas, etc.)
├── requirements_runtime.txt           ← Lightweight deps (streamlit, pinecone, openai)
├── .env.example                       ← All required env vars with comments
├── .gitignore
│
├── data/
│   ├── raw/                           ← Place PDF source documents here
│   └── processed/
│       └── all_chunks.json            ← Chunk snapshot (written by run_ingestion.py)
│
├── ingestion/                         ← INGESTION ONLY — never imported at runtime
│   ├── run_ingestion.py               ← Single entry point: python -m ingestion.run_ingestion
│   ├── document_loader.py             ← pdfplumber PDF extraction
│   ├── chunker.py                     ← Domain-adaptive chunking
│   └── embedder.py                    ← sentence-transformers EmbeddingModel singleton
│
├── retrieval/                         ← RUNTIME SAFE
│   ├── bm25_index.pkl                 ← Written by run_ingestion.py; loaded at runtime
│   ├── bm25_retriever.py              ← Loads .pkl, BM25 search
│   ├── dense_retriever.py             ← Pinecone Inference API query embedding + search
│   └── rrf_fusion.py                  ← RRF fusion → list[RetrievedChunk]
│
├── generation/                        ← RUNTIME SAFE
│   ├── prompts.py                     ← System prompt template, context formatting
│   └── generator.py                   ← GPT-4o-mini, RAGResponse, greeting short-circuit
│
├── evaluation/                        ← EVALUATION ONLY (not deployed)
│   ├── ragas_eval.py                  ← Faithfulness, answer_relevancy, recall@5, MRR
│   ├── test_questions.txt             ← 30 realistic licensing questions
│   └── results/                       ← Timestamped CSV outputs
│
├── app/
│   └── streamlit_app.py               ← Streamlit chat UI
│
└── tests/
    └── test_retrieval.py              ← Unit tests including separation enforcement
```

---

## Chunking Strategy

Three strategies, dispatched by filename heuristics:

| Strategy | Trigger | Logic |
|----------|---------|-------|
| `section_chunker` | Default (legislation, guidance) | Group by heading level ≤ 2; split oversized sections at level 3 |
| `qa_pair_chunker` | Filename contains "faq", "interview", "question" | Extract Q&A pairs; BM25 indexes question only |
| `fault_block_chunker` | Filename contains "enforcement", "case study", "scenario" | Split on CASE:/SCENARIO:/INCIDENT: delimiters |

**Chunk size limits** (from `.env`):
- `MAX_CHUNK_TOKENS = 600` (whitespace-token count)
- `MIN_CHUNK_TOKENS = 80` (merged into predecessor if smaller)

**Metadata per chunk:**
- `chunk_id` — deterministic slug: `{source}-{index:04d}-{heading_slug}`
- `source_document` — PDF filename stem
- `section_number` — extracted section/paragraph number
- `page_number` — 1-based page where chunk begins
- `chunk_index` — position within document
- `topic_category` — one of 10 licensing topic categories
- `chunking_method` — strategy that produced the chunk
- `token_count` — approximate whitespace-token count

---

## BM25 Implementation

```python
BM25Okapi(corpus, k1=1.5, b=0.75)   # Robertson & Zaragoza 2009 defaults
```

- **No stemming** — preserves section numbers, Act citations (e.g. "s.153")
- **Stopword removal only**
- Indexes `bm25_text` (may differ from `text` — e.g. heading prepended for spec chunks)
- Over-fetches by 3× before filtering: `top_k_bm25 * 3`
- Serialised with `pickle` to `retrieval/bm25_index.pkl`

---

## Pinecone Setup

| Parameter | Value | Env var |
|-----------|-------|---------|
| Index name | `hospitality-compliance` | `PINECONE_INDEX_NAME` |
| Namespace | `hospitality-kb` | `PINECONE_NAMESPACE` |
| Dimension | `1024` | `PINECONE_DIMENSION` |
| Metric | `cosine` | `PINECONE_METRIC` |
| Cloud | `aws` | `PINECONE_CLOUD` |
| Region | `us-east-1` | `PINECONE_REGION` |
| Embed model | `multilingual-e5-large` | `PINECONE_EMBED_MODEL` |

**Note:** chunk `text` is NOT stored in Pinecone metadata. At runtime, chunk text
is resolved from the in-memory BM25 index using the vector ID. The BM25 index and
Pinecone index must always be in sync.

---

## Reciprocal Rank Fusion

```
rrf_score(chunk) = Σ  1 / (k + rank)
                  sources
```

- `k = 60` (Robertson 2009 standard; set via `RRF_K` env var)
- `TOP_K_BM25 = 7`, `TOP_K_DENSE = 7`, `TOP_K_FINAL = 5`

---

## GPT-4o-mini Integration

```python
OPENAI_MODEL    = "gpt-4o-mini"
TEMPERATURE     = 0.1    # factual, grounded
MAX_TOKENS      = 768
```

- **Greeting short-circuit:** regex detects greetings → canned response, no API call
- **Safety detection:** regex detects review/revocation/court queries → injects safety addendum
- **Fallback detection:** checks for "I do not have sufficient information" in answer
- **Citation extraction:** `\[[^\[\]]{5,120}\]` over the answer text

---

## Jurisdiction

**England and Wales only** under the Licensing Act 2003.
The system prompt explicitly notes where Scotland (Licensing (Scotland) Act 2005)
or Northern Ireland rules may differ.

---

## RAGAS Evaluation Targets

```python
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
```

---

## Deployment Checklist

Before deploying to Streamlit Cloud:

- [ ] Run `python -m ingestion.run_ingestion` and confirm `retrieval/bm25_index.pkl` exists
- [ ] Upload `bm25_index.pkl` to a storage bucket OR commit it (check size)
- [ ] Set all secrets in Streamlit Cloud secrets UI (from `.env.example`)
- [ ] Install from `requirements_runtime.txt` only — NOT `requirements_ingestion.txt`
- [ ] Verify `pytest tests/test_retrieval.py::TestIngestionRuntimeSeparation` passes
- [ ] Rotate `OPENAI_API_KEY` and `PINECONE_API_KEY` from `.env` before any public commit

---

## Known Limitations

1. **Approximate token counting** — `len(text.split())` not BPE. GPT-4o-mini may
   see ~20–30% more tokens than estimated near chunk boundaries.
2. **BM25 lookup O(n) per query** — chunk text resolved by iterating all chunks.
   Cache the lookup dict at `HybridRetriever` init for production at scale.
3. **Pinecone/BM25 sync requirement** — re-ingesting to either without the other
   breaks retrieval silently. Always run full ingestion pipeline together.
4. **RAGAS double-retrieval** — evaluation retrieves contexts twice per query.
   Results may differ slightly if retrieval is non-deterministic.
5. **No rate limiting** — add per-IP limits before public deployment.
