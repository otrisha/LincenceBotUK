# CLAUDE.md — Benamdaj Hybrid RAG System
**BDS-RAG-001 | MSc AI & Data Science, University of Wolverhampton**
**Author: Patricia Orji**

> This file documents every design decision, component, and configuration in the
> Benamdaj Dredging Solutions Ltd. Hybrid RAG prototype. It is intended as the
> authoritative reference for future development, the production Streamlit port,
> and thesis examination.

---

## Architecture

### System Overview

This is a **Hybrid Retrieval-Augmented Generation (RAG)** customer support system
built for Benamdaj Dredging Solutions Ltd., a marine engineering SME in Nigeria.
The system answers customer questions about four dredger models using the company's
own technical documentation as the exclusive knowledge source.

The architecture has two completely separate phases:

```
INGESTION PHASE (run once, offline)
────────────────────────────────────────────────────────────────
 PDF Documents
     │
     ▼
 document_loader.py     ← pdfplumber extracts blocks (headings, paragraphs, tables)
     │
     ▼
 chunker.py             ← 4 domain-adaptive strategies produce Chunk objects
     │
     ├──▶ BM25Index      ← serialised to data/bm25_index.pkl via pickle
     │
     └──▶ EmbeddingModel ← sentence-transformers encodes chunks → 384-dim vectors
                              │
                              ▼
                          Pinecone upsert (namespace: benamdaj-kb)

RUNTIME PHASE (every user query)
────────────────────────────────────────────────────────────────
 User Query
     │
     ▼
 query_processor.py     ← cleans text, detects model, classifies topic,
     │                     builds Pinecone metadata filter
     │
     ▼
 hybrid_retriever.py    ← runs BM25 and Pinecone in parallel, fuses with RRF
     │
     ▼
 prompt_builder.py      ← formats retrieved chunks into numbered context passages
     │
     ▼
 generator.py           ← calls OpenAI gpt-4o-mini, verifies citations, detects fallback
     │
     ▼
 Flask /chat endpoint   ← returns JSON {answer, sources, is_fallback, latency, ...}
```

### How All Components Connect

| Component | File | Calls | Called By |
|-----------|------|-------|-----------|
| Flask app | `main.py` | `query_processor`, `hybrid_retriever`, `generator` | Browser / CLI |
| Query processor | `retrieval/query_processor.py` | `utils/helpers.clean_text` | `main.py`, evaluator |
| Hybrid retriever | `retrieval/hybrid_retriever.py` | `BM25Index`, `EmbeddingModel`, `Pinecone` | `main.py`, evaluator |
| Generator | `generation/generator.py` | `prompt_builder`, `OpenAI` client | `main.py`, evaluator |
| Prompt builder | `generation/prompt_builder.py` | — | `generator` |
| BM25 index | `ingestion/indexer.BM25Index` | `rank_bm25.BM25Okapi`, `nltk` | `hybrid_retriever`, evaluator |
| Embedding model | `ingestion/indexer.EmbeddingModel` | `sentence_transformers` | `indexer`, `hybrid_retriever` |
| Document loader | `ingestion/document_loader.py` | `pdfplumber` | `ingest.py` |
| Chunker | `ingestion/chunker.py` | `document_loader`, `utils/helpers` | `ingest.py` |
| Evaluator | `evaluation/ragas_evaluator.py` | everything above + RAGAS | standalone |
| Settings | `config/settings.py` | `python-dotenv` | every module |
| Prompts | `config/prompts.py` | — | `generation/generator.py` (partially) |

**Note:** `config/prompts.py` defines a more detailed `SYSTEM_PROMPT` with inline
document IDs and company context. The actual runtime uses `generation/prompt_builder.py`
which has its own `SYSTEM_PROMPT_TEMPLATE`. Both files exist; `prompt_builder.py` is
what the live generator calls.

---

## Libraries & Dependencies

All pins are in `requirements.txt`.

| Library | Version Pin | Purpose | Memory Profile |
|---------|-------------|---------|----------------|
| `rank-bm25` | `==0.2.2` | BM25Okapi sparse retrieval index | Lightweight — pure Python, in-memory token arrays |
| `sentence-transformers` | `>=2.7,<3.0` | Loads `all-MiniLM-L6-v2` embedding model | **Heavy** — ~90 MB model weights loaded into RAM on startup |
| `nltk` | `>=3.8` | Tokenisation (`word_tokenize`) and stopword removal for BM25 | Medium — downloads `punkt`, `punkt_tab`, `stopwords` data on first run |
| `pinecone` | `>=4.0,<6.0` | Serverless vector database client; upsert and query | Lightweight client (network I/O only) |
| `openai` | `>=1.30,<2.0` | OpenAI Python SDK for gpt-4o-mini chat completions | Lightweight client (network I/O only) |
| `pdfplumber` | `>=0.10` | PDF text + table extraction with font-size and bounding-box metadata | Medium — holds PDF page in memory during extraction |
| `langchain` | `>=0.2,<0.4` | Required as a dependency by `ragas` (not used directly in pipeline) | Medium |
| `langchain-openai` | `>=0.1,<0.3` | `ChatOpenAI` wrapper used by RAGAS evaluation runner | Medium |
| `langchain-community` | `>=0.2,<0.4` | Transitive RAGAS dependency | Medium |
| `ragas` | `>=0.1,<0.3` | Faithfulness, answer relevancy, context precision/recall metrics | **Heavy** — pulls in LangChain stack; evaluation only |
| `datasets` | `>=2.18` | `Dataset.from_dict()` for RAGAS input format | Medium |
| `Flask` | `>=3.0` | Web server and REST API (`/`, `/chat`, `/health`) | Lightweight |
| `numpy` | `>=1.26` | Float32 vector arrays for embeddings | Medium |
| `pandas` | `>=2.2` | Evaluation results DataFrame and CSV export | Medium |
| `python-dotenv` | `>=1.0` | Loads `config/.env` into `os.environ` | Lightweight |
| `tqdm` | `>=4.66` | Progress bars during chunk encoding and evaluation | Lightweight |
| `pytest` | `>=8.0` | Unit test runner | Dev-only |

**Heavy at runtime (models loaded into RAM):**
- `sentence-transformers/all-MiniLM-L6-v2` — ~90 MB, loaded once as singleton
- `BM25Index` — loaded from `.pkl`, size proportional to corpus (~few MB)

**Heavy at evaluation time only (not loaded during normal chat):**
- `ragas`, `langchain`, `langchain-openai`, `datasets`

---

## Chunking Strategy

### Overview

Chunking is **domain-adaptive**: each source document uses a different strategy matched
to its content type. This is a deliberate thesis design decision — generic fixed-size
chunking loses procedural and Q&A structure, which is critical for accurate retrieval
in a technical domain.

### Chunk Size Limits

Defined in `config/settings.py`:

```python
MAX_CHUNK_TOKENS = 600   # split if section text exceeds this
MIN_CHUNK_TOKENS = 80    # merge into previous chunk if smaller than this
```

Token count is approximated by `utils/helpers.count_tokens()` as `len(text.split())` —
a whitespace-split word count, not a true BPE token count. This is a prototype
simplification; real token counts (via `tiktoken`) would differ by ~20%.

### The Four Strategies

#### Strategy 1: `section_chunker` — BDS-SPEC-001 (Product Specification Manual)

- Groups PDF blocks by heading level ≤ 2 using `_group_by_heading()`
- If a section exceeds `MAX_CHUNK_TOKENS`, recursively splits by heading level 3
- Each chunk's `bm25_text` is `"{heading} {body_text}"` (heading prepended for
  keyword matching boost)
- `model` field is auto-detected from text via `_detect_model_from_text()`
- All chunks get `retrieval_priority="high"`

#### Strategy 2: `procedure_chunker` — BDS-OM-001 (Operations & Maintenance Manual)

- Groups by heading level ≤ 2, then within each section scans for procedure-
  related headings matching `_PROC_RE` regex (startup, shutdown, maintenance, etc.)
- Each procedure-level heading becomes a chunk boundary
- Falls back to `section_chunker` if no procedure headings are detected
- All chunks get `model="All"` (procedures apply to all dredger models)

#### Strategy 3: `fault_block_chunker` — BDS-TSG-001 (Troubleshooting Guide)

- Ignores heading structure; uses regex to split on `SYMPTOM:` delimiters:
  ```python
  re.split(r"(?=SYMPTOM\s*:)", raw, flags=re.I)
  ```
- Each fault block becomes one chunk
- First part (before any SYMPTOM) becomes the "Quick Symptom Reference Index" chunk
- `model` auto-detected per block; `topic_category="fault_diagnosis"`

#### Strategy 4: `qa_pair_chunker` — BDS-FAQ-001 (Staff Interview FAQs)

- Extracts Q&A pairs using:
  ```python
  re.compile(r"Q\s*\n(.+?)\n\s*A\s*\n(.+?)(?=\nQ\s*\n|\Z)", re.DOTALL | re.IGNORECASE)
  ```
- Each Q&A pair becomes one chunk
- `bm25_text` is `"{question_text} {topic_category}"` — answer is excluded from BM25
  tokens so keyword search hits the question, not the answer prose
- `topic_category` classified by keyword scoring against 8 categories
- `persona_role` extracted from 300 characters before the Q marker using `_PERSONA_MAP`
- `retrieval_priority="high"` for fault, operations, and maintenance topics; `"medium"` otherwise

### Metadata Attached to Every Chunk

Every `Chunk` dataclass carries:

```python
chunk_id           # e.g. "BDS-FAQ-001-023-why-does-production-drop"
document_id        # "BDS-SPEC-001" | "BDS-OM-001" | "BDS-TSG-001" | "BDS-FAQ-001" | "BDS-PL-001"
document_title     # human-readable title
source             # "product_spec" | "om_manual" | "troubleshooting" | "staff_faq" | "price_list"
knowledge_type     # "explicit_structured" | "explicit_procedural" | "explicit_diagnostic" | "tacit_elicited"
chunking_method    # "section" | "procedure" | "fault_block" | "qa_pair"
text               # full text shown to the LLM
bm25_text          # (sometimes different) text indexed by BM25 — heading-boosted or question-only
heading            # section / symptom / FAQ heading
model              # "Model 1" | "Model 2" | "Model 3" | "Model 4" | "All"
topic_category     # one of 8 categories
persona_role       # staff persona (FAQ only)
retrieval_priority # "high" | "medium"
token_count        # whitespace-token count of `text`
```

The `text` vs `bm25_text` split is intentional: BM25 indexes a curated signal,
while the LLM sees the full passage.

### Short-Chunk Merging

After any chunker runs, `_merge_short()` merges any chunk below `MIN_CHUNK_TOKENS`
(80 tokens) into its predecessor. This prevents degenerate single-sentence chunks.

### Chunk ID Format

Generated by `utils/helpers.generate_chunk_id()`:
```
{document_id}-{index:03d}-{slugified_label[:40]}
```
Example: `BDS-FAQ-001-023-why-does-production-drop-in-the-af`

---

## BM25 Implementation

### How the Index is Built

File: `ingestion/indexer.py`, class `BM25Index`

```python
BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
# k1 = 1.5  (Robertson & Zaragoza 2009 defaults)
# b  = 0.75
```

The corpus is a list of token lists, one per chunk. Tokens come from `tokenise_for_bm25()`:

```python
def tokenise_for_bm25(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in _STOPWORDS]
```

Key design choices:
- **No stemming** — deliberate. Technical terms like `3408`, `16/14`, `rpm` must
  match exactly. Stemming would corrupt these.
- **Stopword removal only** — reduces noise without damaging technical vocabulary.
- **Indexes `bm25_text`, not `text`** — the chunker controls what BM25 sees. For
  FAQ chunks, only the question is indexed; for spec chunks, the heading is prepended.

The `BM25Index` object stores the list of `Chunk` objects in the same positional
order as the BM25 corpus. Score index `i` maps directly to `chunks[i]`. This alignment
is critical and must be preserved if the index is ever rebuilt.

### How the Index is Serialised

```python
def save(self, path: Path = BM25_INDEX_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(self, f)    # pickles the entire BM25Index object
```

Saved to `data/bm25_index.pkl`. The entire object — BM25Okapi model + chunk list —
is pickled together. This means the same Python version and `rank-bm25` version
**must** be used when loading.

### How it is Queried at Runtime

```python
def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
    tokens = tokenise_for_bm25(query)      # same tokeniser as indexing
    scores = self._bm25.get_scores(tokens) # BM25Okapi.get_scores()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in ranked[:top_k] if score > 0]
```

Zero-scoring chunks are excluded (chunks with no term overlap are irrelevant).

In `hybrid_retriever._bm25_search()`, the raw BM25 results are post-filtered:
- Model filter: if a specific model is detected, chunks with `model` not in
  `{detected_model, "All"}` are dropped
- FAQ exclusion: when `mode="hybrid_no_faq"`, `source="staff_faq"` chunks are skipped
- Over-fetches by 3× (`top_k * 3`) before filtering to ensure enough results survive

### Where BM25 Lives in the Pipeline

- **Ingestion**: built in `ingest.py` → `BM25Index(chunks)` → `bm25.save()`
- **Runtime**: loaded in `main.get_retriever()` → `BM25Index.load()` → passed to
  `HybridRetriever.__init__(bm25_index)`
- **Query**: called in `HybridRetriever.retrieve()` → `self._bm25_search()`

---

## Pinecone Setup

| Parameter | Value | Source |
|-----------|-------|--------|
| Index name | `benamdaj-rag` | `PINECONE_INDEX_NAME` in settings / `.env` |
| Namespace | `benamdaj-kb` | `PINECONE_NAMESPACE` in settings |
| Dimensions | `384` | `PINECONE_DIMENSION` — matches `all-MiniLM-L6-v2` output |
| Distance metric | `cosine` | `PINECONE_METRIC` |
| Cloud | `aws` | `PINECONE_CLOUD` |
| Region | `us-east-1` | `PINECONE_REGION` |
| Spec type | Serverless | `ServerlessSpec` in `indexer._ensure_pinecone_index()` |

### How Vectors are Uploaded

`ingestion/indexer.upsert_to_pinecone()`:

1. Instantiates `Pinecone(api_key=PINECONE_API_KEY)`
2. Calls `_ensure_pinecone_index()` — creates the index if it doesn't exist,
   polling every 2 seconds up to 60 seconds until `status.ready == True`
3. Builds a list of vector dicts:
   ```python
   {"id": chunk.chunk_id, "values": vec.tolist(), "metadata": {...}}
   ```
4. Uploads in batches of 100 vectors (default `batch_size=100`)

Metadata stored per vector (used for filtering at query time):
```python
{
    "document_id"       : chunk.document_id,
    "source"            : chunk.source,
    "knowledge_type"    : chunk.knowledge_type,
    "model"             : chunk.model,
    "topic_category"    : chunk.topic_category,
    "heading"           : chunk.heading[:200],   # truncated to 200 chars
    "retrieval_priority": chunk.retrieval_priority,
    "token_count"       : chunk.token_count,
    "persona_role"      : chunk.persona_role,
}
```

**Note:** `text` is NOT stored in Pinecone metadata. At runtime, retrieved chunk
text is looked up from the in-memory `BM25Index` chunk list using the vector ID.
This is a critical coupling: the BM25 index must contain every chunk that was
uploaded to Pinecone, with matching `chunk_id` values.

### How Similarity Search is Performed

`hybrid_retriever._dense_search()`:

```python
self._pinecone().query(
    vector    = vec.tolist(),    # 384-dim float32 query embedding
    top_k     = TOP_K_DENSE,    # 7 candidates
    include_metadata = True,
    namespace = "benamdaj-kb",
    filter    = eff_filter,      # optional metadata filter
)
```

When a model is detected, the filter is:
```python
{"$or": [{"model": {"$eq": "Model 1"}}, {"model": {"$eq": "All"}}]}
```
When `mode="hybrid_no_faq"`, an additional filter excludes `source: staff_faq`.

The Pinecone client is lazily initialised and cached as `self._pc_index` on first call.

---

## Embedding Model

| Parameter | Value |
|-----------|-------|
| Model name | `sentence-transformers/all-MiniLM-L6-v2` |
| Output dimensions | `384` |
| Batch size (encoding) | `32` |
| Normalisation | `normalize_embeddings=True` (L2 norm — required for cosine metric) |
| Output dtype | `np.float32` |

### Where and How it is Loaded

`ingestion/indexer.EmbeddingModel` is a **singleton** implemented with a class variable:

```python
class EmbeddingModel:
    _instance: SentenceTransformer | None = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._instance is None:
            cls._instance = SentenceTransformer(EMBEDDING_MODEL)
        return cls._instance
```

The model is downloaded from HuggingFace Hub on first call and cached locally
(default HuggingFace cache at `~/.cache/huggingface/`). Subsequent calls return
the already-loaded instance.

The singleton is **loaded at application startup** in `main.get_retriever()`:
```python
EmbeddingModel.get()   # line 80 of main.py
```

This means the ~90 MB model is loaded into RAM every time the Flask server or
evaluator starts — even if only BM25 queries are being served.

### Where it is Called in the Pipeline

- **Ingestion**: `EmbeddingModel.encode_chunks(chunks)` — encodes all chunk texts
  in batches of 32, returns `np.ndarray` of shape `(n_chunks, 384)`
- **Runtime query**: `EmbeddingModel.encode_query(pq.cleaned_query)` in
  `hybrid_retriever.retrieve()` — encodes the single query string, returns shape `(384,)`

---

## Reciprocal Rank Fusion

### The Exact Formula

```python
rrf_score(chunk_id) = Σ  1 / (k + rank)
                     sources
```

Where `k = RRF_K = 60` (the Robertson 2009 smoothing constant, set in `config/settings.py`).

### How BM25 and Pinecone Results are Combined

`retrieval/hybrid_retriever._rrf()`:

```python
for rank, (chunk, _) in enumerate(zip(bm25_chunks, bm25_results), start=1):
    rrf_scores[cid] += 1.0 / (k + rank)    # BM25 contribution

for rank, (cid, _) in enumerate(dense_results, start=1):
    rrf_scores[cid] += 1.0 / (k + rank)    # Dense contribution
```

A chunk appearing in **both** lists accumulates scores from both sources.
A chunk appearing in only one list still scores. Chunks absent from both
lists score zero and are excluded.

The merged dictionary is then sorted descending by score and the top
`TOP_K_FINAL = 5` results are returned.

### The k Constant

`RRF_K = 60` — the standard value from Robertson & Zaragoza (2009). Higher k
reduces the penalty for low-ranked results (more uniform blending). The value
is not tuned in this prototype; 60 is the accepted default.

### Input Sizes

```python
TOP_K_BM25  = 7   # BM25 candidates fed into RRF
TOP_K_DENSE = 7   # Pinecone candidates fed into RRF
TOP_K_FINAL = 5   # chunks passed to the LLM after fusion
```

In the worst case (all 14 candidates are unique), RRF produces 14 candidates
from which the top 5 are selected. In the best case (perfect overlap), 7
candidates are scored from two sources each and the top 5 are highly confident.

---

## GPT-4o-mini Integration

### How the Client is Initialised

`generation/generator.py`:

```python
_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client
```

Module-level lazy singleton. `OPENAI_API_KEY` is read from `config/settings.py`
which loads it from `config/.env` via `python-dotenv`.

### The Full Prompt Structure

The actual runtime prompt is built in `generation/prompt_builder.py`:

**System prompt** (`SYSTEM_PROMPT_TEMPLATE`):
```
You are a technical customer support assistant for Benamdaj Dredging Solutions Ltd.,
a marine engineering company based in Nigeria. You assist customers with questions
about four dredger models:

  - Model 1: 16/14 Inch Cutter Suction Dredger       (BDS-CSD-1614)
  - Model 2: 14/12 Inch Amphibious Multifunctional    (BDS-AMD-1412)
  - Model 3: 12/10 Inch Bucket Chain Dredger          (BDS-BCD-1210)
  - Model 4: 10/10 Inch Jet Suction Dredger           (BDS-JSD-1010)

INSTRUCTIONS — follow these strictly:
1. BASE YOUR ANSWER ONLY on the retrieved context passages provided below.
2. CITE YOUR SOURCES. After every factual claim, include [BDS-SPEC-001] etc.
3. IF CONTEXT IS INSUFFICIENT: respond with exactly "I do not have that information..."
4. MODEL SPECIFICITY. Restrict to the queried model; state cross-model differences.
5. SAFETY QUERIES. Recommend BDS-OM-001 and qualified engineers.
6. TONE. Be clear, professional, and appropriately technical.
{safety_addendum}   ← inserted only when is_safety_query=True
---
RETRIEVED CONTEXT PASSAGES:
{context_passages}  ← numbered blocks with source, section, model metadata headers
---
```

**User message**: the raw `pq.cleaned_query` string (no additional wrapping).

Context passages are formatted by `format_context_passages()`:
```
[Passage 1 | Source: BDS-FAQ-001 | Section: FAQ-07: What realistic... | Model: Model 1]
<chunk.text>

---

[Passage 2 | ...]
<chunk.text>
```

**Safety addendum** (injected when `is_safety_query=True`):
```
IMPORTANT — SAFETY-CRITICAL QUERY DETECTED: Provide accurate documentation
information and strongly recommend direct contact with a qualified Benamdaj
engineer before taking any action.
```

### Generation Settings

```python
OPENAI_MODEL  = "gpt-4o-mini"   # read from .env; defaults to gpt-4o-mini
TEMPERATURE   = 0.1              # low temperature → factual, grounded responses
MAX_TOKENS    = 768              # max output tokens
```

The API call:
```python
_get_client().chat.completions.create(
    model       = OPENAI_MODEL,
    temperature = TEMPERATURE,
    max_tokens  = MAX_TOKENS,
    messages    = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},   # cleaned_query only
    ],
)
```

### Post-Processing

After the API call, `generator.generate()`:
1. Detects fallback: checks if `"I do not have that information"` appears in the answer
2. Extracts citations: regex `\[BDS-(?:SPEC|OM|TSG|FAQ|PL)-\d+\]` over the answer text
3. Logs a warning if no citations are present and it's not a fallback
4. Records latency via `time.perf_counter()`

### Greeting Short-Circuit

When `pq.is_greeting=True`, the generator returns a canned response immediately
without calling BM25, Pinecone, or OpenAI. This avoids unnecessary API costs for
greetings.

---

## RAGAS Evaluation

### Which RAGAS Metrics are Used

```python
from ragas.metrics import (
    faithfulness,        # are claims in answer supported by context?
    answer_relevancy,    # does the answer address the question?
    context_precision,   # are retrieved contexts ranked well relative to relevance?
    context_recall,      # does context cover the ground truth answer?
)
```

Additionally, three retrieval metrics are computed without RAGAS:
- **Recall@5** — does the relevant document appear in the top-5 retrieved chunks?
- **Precision@5** — what fraction of retrieved chunks are from the relevant document?
- **MRR (Mean Reciprocal Rank)** — reciprocal of the rank of the first relevant chunk

### Evaluation Dataset Structure

`evaluation/eval_queries.py` defines 120 `EvalQuery` objects (as of the final
commit; README says 100 — the code has the authoritative count):

```python
@dataclass
class EvalQuery:
    query_id      : str    # e.g. "FAQ-07"
    query         : str    # the natural language question
    expected_model: str    # "Model 1"|"Model 2"|"Model 3"|"Model 4"|"All"
    topic_category: str    # one of 8 categories
    relevant_doc  : str    # primary document_id (e.g. "BDS-FAQ-001")
    ground_truth  : str    # reference answer for RAGAS context_recall
```

Distribution across documents:
- BDS-SPEC-001: 20 queries (spec lookups across 4 models + cross-model)
- BDS-OM-001: 17 queries (startup, shutdown, maintenance, safety)
- BDS-TSG-001: 12 queries (fault diagnosis)
- BDS-FAQ-001: 18 queries (staff knowledge, tacit expertise)
- BDS-PL-001: 20 queries (pricing, commercial)
- Cross-document: 5 queries requiring synthesis

### How Scores are Generated and Stored

`evaluation/ragas_evaluator.run_evaluation()`:

1. Loads BM25 index and embedding model
2. For each query: processes → retrieves → generates → records metrics
3. Re-retrieves contexts for all queries (second pass) for RAGAS input
4. Calls `_run_ragas()` which wraps `ragas.evaluate()` with `ChatOpenAI(gpt-4o-mini)`
5. Saves results as timestamped CSV: `logs/evaluation/eval_{mode}_{YYYYMMDD_HHMMSS}.csv`

Existing evaluation runs in `logs/evaluation/`:
- `eval_hybrid_20260409_193227.csv`
- `eval_hybrid_20260410_102355.csv`
- `eval_hybrid_20260410_113053.csv`
- `eval_hybrid_20260410_115443.csv`
- `eval_hybrid_20260410_120010.csv`
- `eval_hybrid_20260410_120445.csv`

### Evaluation Targets (from `config/settings.py`)

```python
EVAL_TARGETS = {
    "recall_at_5"         : 0.80,
    "precision_at_5"      : 0.40,   # adjusted for 5-doc corpus
    "mrr"                 : 0.70,
    "ragas_faithfulness"  : 0.80,
    "ragas_answer_rel"    : 0.75,
    "ragas_ctx_precision" : 0.65,
    "ragas_ctx_recall"    : 0.75,
    "max_latency_seconds" : 8.0,
}
```

### Ablation Configurations

Four modes for the ablation study (Thesis Chapter 8):

| Mode | Config | Description |
|------|--------|-------------|
| `bm25_only` | A | Sparse baseline — BM25 scores converted to pseudo-RRF ranks |
| `dense_only` | B | Dense baseline — Pinecone cosine scores only |
| `hybrid` | C | Proposed system — BM25 + Dense + RRF (default) |
| `hybrid_no_faq` | D | Hybrid without BDS-FAQ-001 — tests tacit knowledge contribution |

---

## File Structure

```
Thesis/
├── config/
│   ├── .env                 ← API keys (gitignored; contains live keys — see WARNING below)
│   ├── __init__.py
│   ├── prompts.py           ← Detailed system prompt + context/fallback templates (partially used)
│   └── settings.py          ← All constants: API keys, model names, index config, chunking limits
│
├── data/
│   ├── bm25_index.pkl       ← Pickled BM25Index object (produced by ingest.py)
│   ├── chunks/
│   │   └── all_chunks.json  ← All chunks serialised for inspection / RAGAS context
│   └── documents/           ← Source PDFs (5 files)
│       ├── BDS-FAQ-001_Staff_Interview_FAQs.pdf
│       ├── Benamdaj_OM_Manual.pdf
│       ├── Benamdaj_price_list.pdf
│       ├── Benamdaj_Product_Specification_Manual.pdf
│       └── Benamdaj_Troubleshooting_Guide.pdf
│
├── evaluation/
│   ├── __init__.py
│   ├── eval_queries.py      ← 120 EvalQuery objects with ground truths; balanced_sample()
│   └── ragas_evaluator.py   ← Recall@5, Precision@5, MRR + RAGAS; ablation runner; CSV export
│
├── generation/
│   ├── __init__.py
│   ├── generator.py         ← OpenAI client singleton; generate(); greeting short-circuit; RAGResponse
│   └── prompt_builder.py    ← SYSTEM_PROMPT_TEMPLATE; format_context_passages(); build_prompt()
│
├── ingestion/
│   ├── __init__.py
│   ├── chunker.py           ← Chunk dataclass; 4 chunking strategies; _merge_short(); dispatch
│   ├── document_loader.py   ← pdfplumber extraction; DocumentBlock; LoadedDocument; heading inference
│   └── indexer.py           ← BM25Index (build/save/load/search); EmbeddingModel singleton; upsert_to_pinecone()
│
├── logs/
│   ├── benamdaj_rag.log     ← Rotating log from all modules (colour-stripped)
│   └── evaluation/          ← Timestamped CSV files from each evaluation run
│
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_retriever.py  ← HybridRetriever; _rrf(); _dense_search(); _bm25_search(); RetrievedChunk
│   └── query_processor.py   ← ProcessedQuery; clean, detect model, classify topic, build Pinecone filter
│
├── static/
│   ├── scripts/chat.js      ← Frontend JS for the Flask chat UI
│   └── styles/chat.css      ← Chat UI styles
│
├── templates/
│   └── index.html           ← Flask HTML template for the chat interface
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py           ← clean_text(); count_tokens(); generate_chunk_id(); table_to_text(); save_json()
│   └── logger.py            ← ColouredFormatter; get_logger(); dual console+file handler
│
├── .gitignore
├── ingest.py                ← Entry point: load → chunk → save JSON → BM25 → embed → Pinecone
├── main.py                  ← Flask app: /, /chat, /health; CLI interactive mode; _ensure_knowledge_base()
├── README.md
└── requirements.txt
```

### Key Imports per File

| File | Imports From |
|------|-------------|
| `main.py` | `flask`, `config.settings`, `generation.generator`, `ingestion.indexer`, `retrieval.*` |
| `ingest.py` | `config.settings`, `ingestion.*`, `utils.*` |
| `ingestion/indexer.py` | `rank_bm25`, `sentence_transformers`, `nltk`, `pinecone`, `numpy` |
| `ingestion/chunker.py` | `ingestion.document_loader`, `config.settings`, `utils.helpers` |
| `ingestion/document_loader.py` | `pdfplumber`, `config.settings`, `utils.helpers` |
| `retrieval/hybrid_retriever.py` | `ingestion.indexer`, `ingestion.chunker`, `retrieval.query_processor`, `pinecone` |
| `retrieval/query_processor.py` | `config.settings`, `utils.helpers` |
| `generation/generator.py` | `openai`, `config.settings`, `generation.prompt_builder`, `retrieval.*` |
| `generation/prompt_builder.py` | `retrieval.hybrid_retriever` |
| `evaluation/ragas_evaluator.py` | `ragas`, `datasets`, `langchain_openai`, everything above |

### Execution Order of the Pipeline

**Ingestion** (`python ingest.py`):
1. `config/settings.py` loaded (dotenv reads `.env`)
2. `ingestion/document_loader.load_all_documents()` — opens each PDF with pdfplumber
3. `ingestion/chunker.chunk_all_documents()` — applies per-document strategy
4. `utils/helpers.save_json()` — writes `data/chunks/all_chunks.json`
5. `ingestion/indexer.BM25Index(chunks)` — builds BM25 corpus and model
6. `BM25Index.save()` — pickles to `data/bm25_index.pkl`
7. `ingestion/indexer.EmbeddingModel.encode_chunks()` — loads sentence-transformer, encodes all chunks
8. `ingestion/indexer.upsert_to_pinecone()` — creates index if needed, uploads in batches of 100

**Runtime** (`python main.py` → Flask → POST /chat):
1. `main.get_retriever()` → `BM25Index.load()`, `EmbeddingModel.get()`
2. `retrieval/query_processor.process_query(query)` → `ProcessedQuery`
3. `retrieval/hybrid_retriever.HybridRetriever.retrieve(pq)`:
   a. `_bm25_search()` → BM25 scores → filtered list
   b. `EmbeddingModel.encode_query()` → 384-dim vector
   c. `_dense_search()` → Pinecone query → matches
   d. `_rrf()` → fused ranking → top 5 `RetrievedChunk` objects
4. `generation/generator.generate(pq, retrieved)`:
   a. `prompt_builder.build_prompt()` → (system_prompt, user_message)
   b. `OpenAI().chat.completions.create()` → answer string
   c. Citation extraction + fallback detection
5. Flask returns JSON response

---

## Known Limitations

### 1. Ingestion Runs Separately From Runtime — But Not Cleanly Separated

The ingestion and runtime code share the same `requirements.txt`. Critically,
`main.py` contains `_ensure_knowledge_base()` which calls `run_ingest()` if the
BM25 or Pinecone index is missing. This means the first cold start of the Flask
server will trigger a full ingestion — loading pdfplumber, sentence-transformers,
and attempting to build Pinecone — all at web server startup time.

**This is a prototype shortcut.** In production, ingestion must run as a
completely separate offline pipeline, never triggered by the web server.

### 2. Sentence-Transformer Model Loaded at Every Runtime Startup

`EmbeddingModel.get()` is called in `main.get_retriever()` at startup. The ~90 MB
`all-MiniLM-L6-v2` model is always loaded into RAM even in `bm25_only` mode.
This is unnecessary in production where query embedding should use Pinecone's
inference API, avoiding any local model at runtime.

### 3. Text is NOT Stored in Pinecone — Tight Coupling Between BM25 and Pinecone

Chunk text is retrieved from the in-memory `BM25Index` using the vector ID as a
key. If Pinecone returns a vector ID that does not exist in the BM25 index,
the chunk is silently dropped with a warning. This means the BM25 index and
Pinecone index must always be in sync — re-uploading to Pinecone without
rebuilding the BM25 index (or vice versa) breaks retrieval.

### 4. API Keys Committed to the Repository

`config/.env` is in `.gitignore` but contains live API keys (OpenAI and Pinecone).
The `.env` file itself is not tracked, but it exists on disk with real credentials.
In production, secrets must be managed via Streamlit Secrets or environment
variables injected by the hosting platform — never via a file committed or left
on disk in a public deployment.

**CRITICAL: Rotate the OpenAI and Pinecone API keys in the `.env` file before
any public deployment or sharing of this repository.**

### 5. Token Counting is Approximate

`utils/helpers.count_tokens()` uses `len(text.split())` — a whitespace word count.
GPT-4o-mini uses BPE tokenisation which counts differently (typically 20–30% more
tokens for technical content). `MAX_CHUNK_TOKENS=600` words may correspond to
~700–800 actual tokens. This rarely causes problems for this corpus size but would
fail silently on edge cases near the context window boundary.

### 6. Pinecone Text Lookup via BM25 Index

Dense-only results from Pinecone are looked up by iterating over the entire
`BM25Index` chunk list to build a `{chunk_id: Chunk}` dict on every `retrieve()`
call:

```python
lookup = {
    self.bm25.get_chunk(i).chunk_id: self.bm25.get_chunk(i)
    for i in range(len(self.bm25))   # O(n) on every query
}
```

For a corpus of ~200 chunks this is negligible, but it does not scale.

### 7. `config/prompts.py` is Partially Superseded

`config/prompts.py` contains a more detailed `SYSTEM_PROMPT` with `{context_passages}`
and `{user_query}` placeholders and a `CONTEXT_PASSAGE_TEMPLATE`. The actual runtime
uses `generation/prompt_builder.py` which has its own template. Both files exist,
but `prompts.py` is not imported by `generator.py` or `prompt_builder.py`. It is
dead code — likely written earlier and superseded by `prompt_builder.py`.

### 8. README Inconsistencies

The `README.md` references `.docx` files and `FastAPI / uvicorn` but the actual
implementation uses `.pdf` files and `Flask`. The README is not accurate. This
`CLAUDE.md` reflects the actual code.

### 9. Evaluation Re-Retrieves All Contexts Twice

`ragas_evaluator.run_evaluation()` retrieves for every query twice: once to
generate answers, and again to collect `contexts` for RAGAS input. This doubles
the number of Pinecone queries and embedding calls during evaluation. It also
means RAGAS evaluates slightly different contexts from what the generator saw
(if retrieval is non-deterministic).

### 10. Hardcoded Values That Should Be Environment Variables

| Hardcoded value | Location | Should become |
|----------------|----------|---------------|
| `PINECONE_NAMESPACE = "benamdaj-kb"` | `settings.py` line 39 | `PINECONE_NAMESPACE` env var |
| `PINECONE_DIMENSION = 384` | `settings.py` line 37 | Derived from model or env var |
| `PINECONE_METRIC = "cosine"` | `settings.py` line 38 | `PINECONE_METRIC` env var |
| `BM25_K1 = 1.5`, `BM25_B = 0.75` | `settings.py` lines 58–59 | Tunable via env var |
| `TEMPERATURE = 0.1`, `MAX_TOKENS = 768` | `settings.py` lines 49–50 | Env vars |
| `TOP_K_BM25=7`, `TOP_K_DENSE=7`, `TOP_K_FINAL=5`, `RRF_K=60` | `settings.py` | Env vars |
| `batch_size=100` in `upsert_to_pinecone()` | `indexer.py` line 159 | Config constant |
| `EMBEDDING_BATCH = 32` | `settings.py` | Already a constant; could be env var |

---

## Production Modifications Required

The following changes are **mandatory** before deploying this system as a publicly
accessible Streamlit application. These are not optional polish items — they are
architectural requirements for a safe, scalable, cost-effective deployment.

---

### 1. Separate the Ingestion Pipeline Completely From Runtime

**Current state:** `main.py` calls `run_ingest()` via `_ensure_knowledge_base()`
if the index is missing. The web server and the ingestion pipeline share
`requirements.txt` and `main.py`.

**Required change:**
- Create a standalone `ingest.py` script (already exists) as the **only** ingestion
  entry point. It must never be imported or called by the Streamlit app.
- Remove `_ensure_knowledge_base()` and the `run_ingest` import from the runtime
  entrypoint entirely.
- Document that ingestion is a one-time offline step (or a scheduled CI/CD job)
  and the Streamlit app assumes the index already exists.

---

### 2. Remove Local Model Loading at Runtime

**Current state:** `EmbeddingModel.get()` loads `all-MiniLM-L6-v2` (~90 MB) at
every startup inside the Streamlit worker process. `sentence-transformers` and
`torch` are heavy dependencies that bloat the deployment container and consume RAM.

**Required change:**
- At runtime (Streamlit app), encode the user query using the **Pinecone Inference
  API** instead of a local model:
  ```python
  # Replace EmbeddingModel.encode_query() with:
  pc = Pinecone(api_key=PINECONE_API_KEY)
  result = pc.inference.embed(
      model  = "multilingual-e5-large",   # or whichever hosted model is configured
      inputs = [query],
      parameters = {"input_type": "query"},
  )
  vec = result[0].values
  ```
- This requires re-ingesting with the Pinecone-hosted model to ensure dimension
  and embedding space match. Alternatively, use Pinecone's `llama-text-embed-v2`
  or another hosted model whose output dimension matches the index.
- Remove `sentence-transformers`, `torch`, and `transformers` from the **runtime**
  `requirements.txt` (they belong only in the ingestion requirements file).

---

### 3. Serialise the BM25 Index Before Deployment

**Current state:** The BM25 index is already serialised to `data/bm25_index.pkl`
by `ingest.py`. However, the `.pkl` file is not committed to git (it's large and
binary) and may not be included in the Streamlit deployment package.

**Required change:**
- After ingestion, commit `data/bm25_index.pkl` to the repository **or** upload
  it to a persistent storage location (e.g. cloud bucket) that the Streamlit app
  can fetch at startup.
- Alternatively, host the BM25 index in a storage bucket (S3, GCS) and fetch it
  at Streamlit startup with a one-time download:
  ```python
  @st.cache_resource
  def load_bm25():
      # download from bucket if not local
      return BM25Index.load(BM25_INDEX_PATH)
  ```
- The `.pkl` file is Python-version and `rank-bm25`-version sensitive. Pin both
  exactly in the ingestion environment and use the same versions in deployment.

---

### 4. Replace `.env` Files With Streamlit Secrets

**Current state:** API keys are loaded from `config/.env` via `python-dotenv`
(`load_dotenv()` in `config/settings.py`).

**Required change:**
- In the Streamlit app, remove the `load_dotenv()` call.
- Store all secrets in Streamlit's secrets management:
  ```toml
  # .streamlit/secrets.toml (local dev)
  OPENAI_API_KEY   = "sk-..."
  PINECONE_API_KEY = "pcsk_..."
  PINECONE_INDEX_NAME = "benamdaj-rag"
  PINECONE_NAMESPACE  = "benamdaj-kb"
  ```
- Read them in `settings.py` via `st.secrets`:
  ```python
  import streamlit as st
  OPENAI_API_KEY   = st.secrets["OPENAI_API_KEY"]
  PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
  ```
- In Streamlit Cloud (or other hosting), configure secrets through the platform's
  secrets UI — never commit `.streamlit/secrets.toml` to version control.

---

### 5. Two Separate `requirements.txt` Files

**Current state:** One `requirements.txt` contains both ingestion-heavy packages
(pdfplumber, sentence-transformers, torch) and runtime-only packages (Flask,
openai, pinecone).

**Required change — create two files:**

**`requirements-ingest.txt`** (run offline / in CI):
```
rank-bm25==0.2.2
sentence-transformers>=2.7,<3.0
nltk>=3.8
pinecone>=4.0,<6.0
pdfplumber>=0.10
numpy>=1.26
tqdm>=4.66
python-dotenv>=1.0
ragas>=0.1,<0.3        # evaluation only; can be a third requirements-eval.txt
datasets>=2.18
langchain>=0.2,<0.4
langchain-openai>=0.1,<0.3
langchain-community>=0.2,<0.4
openai>=1.30,<2.0
pandas>=2.2
pytest>=8.0
```

**`requirements-runtime.txt`** (Streamlit deployment):
```
pinecone>=4.0,<6.0        # vector search client only (no local model)
openai>=1.30,<2.0         # LLM API
rank-bm25==0.2.2          # BM25 query at runtime
nltk>=3.8                 # BM25 tokenisation
streamlit>=1.35           # web UI
numpy>=1.26
python-dotenv>=1.0        # local dev only; remove if using st.secrets exclusively
```

Key omissions from runtime: `sentence-transformers`, `torch`, `pdfplumber`,
`ragas`, `datasets`, `langchain*`, `Flask`, `pytest`.

---

### 6. Replace Flask With Streamlit

**Current state:** The UI is a Flask app (`main.py`) with an HTML/CSS/JS chat
interface in `templates/` and `static/`.

**Required change:**
- Create `app.py` as the Streamlit entry point.
- Use `st.cache_resource` for the BM25 index and OpenAI client (loaded once per
  worker):
  ```python
  @st.cache_resource
  def load_retriever():
      bm25 = BM25Index.load()
      return HybridRetriever(bm25, mode="hybrid")
  ```
- Use `st.chat_message` and `st.chat_input` for the conversation UI.
- Remove `templates/`, `static/`, and all Flask-specific code.
- The `/health` endpoint has no equivalent in Streamlit — use Streamlit Cloud's
  built-in health checks or a separate monitoring solution.

---

### 7. Additional Production Hardening

These are not strictly required for Streamlit deployment but are important for
a production-quality system:

- **Rate limiting:** The `/chat` endpoint has no rate limiting. Add per-IP limits
  before public deployment to prevent abuse and runaway API costs.
- **Input length validation:** Add a maximum query character limit (e.g. 500 chars)
  before sending to the embedding model and OpenAI.
- **Async Pinecone client:** The current Pinecone client is synchronous. For
  concurrent Streamlit users, switch to the async client or use `threading` to
  avoid blocking the event loop.
- **BM25 lookup O(n) per query:** The `lookup` dict rebuilt on every call in
  `hybrid_retriever.retrieve()` should be cached at `HybridRetriever.__init__`.
- **RAGAS at evaluation time only:** Ensure `ragas`, `langchain`, and `datasets`
  are never installed in the production environment. They are evaluation tools only.
- **Log rotation:** `utils/logger.py` uses a simple `FileHandler` that will grow
  indefinitely. Replace with `RotatingFileHandler` or a cloud logging sink.
- **`config/.env` API key rotation:** The live API keys currently in `config/.env`
  must be revoked and replaced before any public commit or sharing of the repository.
