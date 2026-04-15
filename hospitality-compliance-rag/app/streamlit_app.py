"""
app/streamlit_app.py
──────────────────────
Streamlit chat interface for the UK Hospitality Licensing Compliance assistant.

Architecture at runtime:
  - BM25 index loaded once from retrieval/bm25_index.pkl via @st.cache_resource
  - Pinecone client initialised once via @st.cache_resource
  - Query embedding handled server-side by the Pinecone Inference API
  - NO sentence-transformers, torch, or heavy ingestion dependencies imported
  - Secrets via st.secrets (Streamlit Cloud) with os.getenv fallback (local dev)

Import chain verification:
  streamlit_app.py
    → retrieval.bm25_retriever    ✓ (pickle + rank_bm25 + nltk only)
    → retrieval.dense_retriever   ✓ (pinecone only — no local model)
    → retrieval.rrf_fusion        ✓ (calls bm25_retriever + dense_retriever)
    → generation.generator        ✓ (openai only)
    → generation.prompts          ✓ (no imports beyond stdlib)
  ✗ ingestion.*                  — never imported by this file

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

# Load .env for local development; in Streamlit Cloud use st.secrets instead
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass   # python-dotenv not in runtime requirements on Cloud


# ---------------------------------------------------------------------------
# Secrets helper — st.secrets first, os.getenv fallback
# ---------------------------------------------------------------------------

def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets (Cloud) or os.environ (local dev)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


# Inject env vars from secrets so downstream modules read them via os.getenv
for _k in (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
    "PINECONE_EMBED_MODEL",
    "PINECONE_DIMENSION",
    "BM25_INDEX_PATH",
    "TOP_K_BM25",
    "TOP_K_DENSE",
    "TOP_K_FINAL",
    "RRF_K",
):
    _val = _secret(_k)
    if _val:
        os.environ[_k] = _val


# ---------------------------------------------------------------------------
# Cached resource loaders
# These run ONCE per Streamlit worker and are reused for every user request.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def _load_bm25():
    """Load BM25 index from disk. Never rebuilt at runtime."""
    from retrieval.bm25_retriever import get_bm25_index
    return get_bm25_index()


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def _load_pinecone():
    """Initialise Pinecone client. Query embedding is server-side."""
    from retrieval.dense_retriever import get_pinecone_client
    return get_pinecone_client()


# Eagerly warm up resources on first load
try:
    _bm25_index = _load_bm25()
    _pc_client  = _load_pinecone()
    _resources_ok = True
    _resource_error = ""
except Exception as exc:
    _resources_ok = False
    _resource_error = str(exc)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UK Licensing Compliance Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚖️ UK Licensing Assistant")
    st.caption("England & Wales | Licensing Act 2003")

    st.markdown("---")

    st.subheader("About")
    st.markdown(
        """
This assistant helps pub landlords, hotel managers, restaurant owners,
and event organisers understand their obligations under UK licensing law.

**Coverage:** England and Wales only, under the Licensing Act 2003.
Scotland and Northern Ireland differences are noted where relevant.
        """
    )

    st.markdown("---")

    st.subheader("How it works")
    st.markdown(
        """
**Hybrid RAG** (Retrieval-Augmented Generation):

1. **BM25 sparse retrieval** — fast keyword matching over licensing documents
2. **Dense semantic retrieval** — Pinecone vector search for meaning-based matching
3. **Reciprocal Rank Fusion** — combines both rankings for the best 5 passages
4. **GPT-4o-mini** — generates a grounded answer from the retrieved passages only

Answers are based exclusively on the loaded licensing documents.
No general knowledge or speculation is used.
        """
    )

    st.markdown("---")

    st.subheader("⚠️ Disclaimer")
    st.warning(
        "This tool provides general guidance only and does not constitute "
        "legal advice. For complex situations — especially licence reviews, "
        "court appeals, or police objections — consult a qualified licensing "
        "solicitor or your local licensing authority.",
        icon="⚠️",
    )

    st.markdown("---")
    st.caption("Built with Hybrid RAG · GPT-4o-mini · Pinecone")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("UK Hospitality Licensing Compliance Assistant")
st.caption(
    "Ask me about premises licences, personal licences, Temporary Event Notices, "
    "licensing objectives, fees, reviews, and more."
)

if not _resources_ok:
    st.error(
        f"**Failed to load knowledge base.** "
        f"Ensure ingestion has been run and the BM25 index exists.\n\n"
        f"Error: {_resource_error}"
    )
    st.stop()

# --- Session state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Source citations", expanded=False):
                for src in msg["sources"]:
                    section_ref = f" § {src['section_number']}" if src.get("section_number") else ""
                    st.markdown(
                        f"- **{src['source_document']}{section_ref}** "
                        f"— {src['heading']} (p. {src['page_number']})"
                    )

# --- Chat input --------------------------------------------------------------
if prompt := st.chat_input("Ask a licensing question..."):
    # Add user message to history and display immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                from retrieval.rrf_fusion import hybrid_retrieve
                from generation.generator import generate

                retrieved = hybrid_retrieve(prompt, bm25_index=_bm25_index)
                response = generate(prompt, retrieved)

                st.markdown(response.answer)

                # Source citations panel
                sources_data = []
                if not response.is_greeting and response.sources:
                    with st.expander(
                        f"Source citations ({len(response.sources)} passages)",
                        expanded=False,
                    ):
                        for chunk in response.sources:
                            section_ref = (
                                f" § {chunk.section_number}"
                                if chunk.section_number else ""
                            )
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(
                                    f"**{chunk.source_document}{section_ref}**  \n"
                                    f"{chunk.heading[:100]}"
                                )
                            with col2:
                                st.caption(f"p. {chunk.page_number}")
                            with st.expander("View passage", expanded=False):
                                st.markdown(chunk.text[:600] + ("..." if len(chunk.text) > 600 else ""))
                            st.divider()

                            sources_data.append({
                                "source_document": chunk.source_document,
                                "section_number":  chunk.section_number,
                                "heading":         chunk.heading,
                                "page_number":     chunk.page_number,
                            })

                # Diagnostic info (collapsed)
                if not response.is_greeting:
                    with st.expander("Diagnostic info", expanded=False):
                        st.caption(
                            f"Model: {response.model} · "
                            f"Latency: {response.latency_seconds:.2f}s · "
                            f"Passages retrieved: {response.retrieved_count} · "
                            f"Fallback: {response.is_fallback} · "
                            f"Safety flag: {response.is_safety}"
                        )

            except Exception as exc:
                error_msg = (
                    f"An error occurred while generating the response: {exc}\n\n"
                    "Please check your API keys and that the knowledge base "
                    "has been ingested."
                )
                st.error(error_msg)
                response = None
                sources_data = []

    # Persist assistant message in history
    if response is not None:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": response.answer,
            "sources": sources_data,
        })

# --- Clear conversation button -----------------------------------------------
if st.session_state.messages:
    if st.button("Clear conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()
