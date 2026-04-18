# UK Hospitality Licensing Compliance Assistant

A **Hybrid Retrieval-Augmented Generation (RAG)** chatbot that answers UK hospitality licensing questions using the Licensing Act 2003 and associated statutory guidance as its exclusive knowledge source.

Built for pub landlords, hotel managers, restaurant owners, event organisers, and licensing applicants. Covers **England and Wales only** under the Licensing Act 2003.

---

## What it does

Ask a plain-English licensing question. The assistant retrieves the most relevant passages from the official documents, passes them to GPT-4o-mini, and returns a grounded, cited answer — without speculating beyond what the documents say.

**Example queries**

- *What are the four licensing objectives?*
- *How many Temporary Event Notices can I apply for per year?*
- *What ID is acceptable under Challenge 25?*
- *What happens if my Designated Premises Supervisor leaves?*
- *What are the mandatory conditions on every alcohol licence?*

---

## Architecture

```
INGESTION (offline, run once)
─────────────────────────────────────────────────────────
 PDF / TXT source documents
       │
       ▼
 document_loader.py     ← pdfplumber extracts text blocks with metadata
       │
       ▼
 chunker.py             ← domain-adaptive chunking (section / QA / case-block)
       │
       ├──▶ chunks.json           ← committed to repo; used as BM25 fallback
       │
       ├──▶ BM25 index            ← serialised to retrieval/bm25_index.pkl
       │
       └──▶ embedder.py           ← Pinecone Inference API (passage embeddings)
                                         │
                                         ▼
                                   Pinecone upsert
                                   (index: hospitality-compliance)

RUNTIME (every query via Streamlit)
─────────────────────────────────────────────────────────
 User question
       │
       ▼
 BM25 sparse retrieval  ← loaded from .pkl (rebuilt from chunks.json if missing)
 Pinecone dense retrieval ← query embedded server-side by Pinecone Inference API
       │
       ▼
 Reciprocal Rank Fusion  ← k=60, top 5 passages
       │
       ▼
 GPT-4o-mini             ← grounded answer with document + section citations
       │
       ▼
 Streamlit chat UI
```

**Key design choices**

| Choice | Reason |
|--------|--------|
| Hybrid BM25 + dense retrieval | Sparse search catches exact legal section numbers; dense search catches semantic matches |
| Pinecone Inference API at runtime | No local embedding model loaded in the deployed app — keeps the container lightweight |
| `chunks.json` committed to repo | Allows BM25 index to be rebuilt at startup on Streamlit Cloud without running ingestion |
| No text stored in Pinecone | Chunk text resolved from BM25 index by vector ID — keeps metadata lean |
| Temperature 0.1 | Grounded, factual answers with minimal hallucination |

---

## Knowledge base

| Document | Type | Coverage |
|----------|------|----------|
| `licensing_act_2003.pdf` | Primary legislation | Full Act — premises licences, personal licences, TENs, reviews, offences |
| `section_182_guidance.pdf` | Statutory guidance | Home Office Section 182 Guidance — authoritative interpretation of the Act |
| `lga_councillors_handbook.pdf` | Guidance | LGA handbook for licensing committees and councillors |
| `challenge_25_guidance.txt` | Guidance | Age verification policy and acceptable ID |
| `ten_overview.txt` | Guidance | Temporary Event Notice procedures and limits |
| `alcohol_licensing_hub.txt` | Guidance | DPS obligations, premises licence conditions, practical compliance |
| `personal_licence_guidance.txt` | Guidance | Personal licence application, qualifications, revocation |

---

## Local setup

### Prerequisites

- Python 3.10+
- A [Pinecone](https://www.pinecone.io/) account (free tier is sufficient for ingestion)
- An [OpenAI](https://platform.openai.com/) API key

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/otrisha/LincenceBotUK.git
cd LincenceBotUK/hospitality-compliance-rag
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=hospitality-compliance
PINECONE_NAMESPACE=hospitality-kb
PINECONE_EMBED_MODEL=multilingual-e5-large
PINECONE_DIMENSION=1024
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### 3. Run ingestion (once)

Ingestion embeds all documents and populates Pinecone. It must be run before the app can answer queries.

```bash
pip install -r requirements_ingestion.txt
python -m ingestion.embedder
```

This will:
- Load and chunk all documents in `data/raw/`
- Embed chunks via the Pinecone Inference API
- Upload 931 vectors to Pinecone
- Build and save the BM25 index to `retrieval/bm25_index.pkl`

> **Note:** The free Pinecone plan has a 250k tokens/minute limit. The ingestion script handles this automatically with retry and rate-limiting between batches.

### 4. Run the app

```bash
pip install -r requirements_runtime.txt
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Streamlit Cloud deployment

The app is designed to deploy directly from this repository with no additional infrastructure.

### Required secrets

In the Streamlit Cloud dashboard, go to **App settings → Secrets** and add:

```toml
OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "pcsk_..."
PINECONE_INDEX_NAME = "hospitality-compliance"
PINECONE_NAMESPACE = "hospitality-kb"
PINECONE_EMBED_MODEL = "multilingual-e5-large"
PINECONE_DIMENSION = "1024"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
```

### How BM25 works on Streamlit Cloud

Streamlit Cloud cannot run the ingestion pipeline. On first startup, if `retrieval/bm25_index.pkl` is not present, the app automatically rebuilds the BM25 index from `data/processed/chunks.json` (which is committed to the repository). This takes approximately 1 second and is cached for the lifetime of the worker process.

Pinecone must be pre-populated by running ingestion locally before deploying.

---

## Project structure

```
hospitality-compliance-rag/
├── app/
│   └── streamlit_app.py          # Streamlit chat UI
├── data/
│   ├── raw/                      # Source documents (PDF and TXT)
│   └── processed/
│       └── chunks.json           # Chunked documents (committed; used by BM25 fallback)
├── evaluation/
│   ├── ragas_eval.py             # RAGAS evaluation pipeline
│   └── test_questions.txt        # 30 evaluation queries
├── generation/
│   ├── generator.py              # GPT-4o-mini client, citation extraction, fallback detection
│   └── prompts.py                # System prompt and context formatting
├── ingestion/
│   ├── chunker.py                # Domain-adaptive chunking strategies
│   ├── document_loader.py        # pdfplumber PDF extraction
│   ├── embedder.py               # Pinecone Inference API embeddings + Pinecone upsert
│   └── run_ingestion.py          # Legacy ingestion entry point
├── retrieval/
│   ├── bm25_retriever.py         # BM25 search (loads pkl; rebuilds from JSON if missing)
│   ├── dense_retriever.py        # Pinecone query embedding and search
│   └── rrf_fusion.py             # Reciprocal Rank Fusion + RetrievedChunk
├── tests/
│   ├── e2e_retrieval_test.py     # End-to-end 6-query retrieval diagnostic
│   └── test_retrieval.py         # Unit tests + ingestion/runtime separation checks
├── .env.example                  # Environment variable template
├── requirements.txt              # Streamlit Cloud deployment dependencies
├── requirements_ingestion.txt    # Ingestion-time dependencies (heavy)
└── requirements_runtime.txt      # Runtime dependencies (lightweight)
```

---

## Evaluation

The system is evaluated against 30 realistic licensing queries covering all major topic areas.

```bash
pip install -r requirements_ingestion.txt
python -m evaluation.ragas_eval --mode hybrid
```

**Metrics computed**

| Metric | Target |
|--------|--------|
| Recall@5 | ≥ 0.80 |
| Precision@5 | ≥ 0.40 |
| MRR | ≥ 0.70 |
| RAGAS Faithfulness | ≥ 0.80 |
| RAGAS Answer Relevancy | ≥ 0.75 |
| RAGAS Context Precision | ≥ 0.65 |
| RAGAS Context Recall | ≥ 0.75 |
| Max latency | ≤ 8.0s |

Results are saved as timestamped CSV files in `evaluation/results/`.

A quick end-to-end diagnostic across 6 representative queries can be run without RAGAS:

```bash
python tests/e2e_retrieval_test.py
```

---

## Disclaimer

This tool provides general guidance only and does not constitute legal advice. For complex situations — especially licence reviews, court appeals, or police objections — consult a qualified licensing solicitor or your local licensing authority.

Coverage is limited to **England and Wales** under the Licensing Act 2003. Scottish and Northern Ireland licensing law differs and is not covered.
