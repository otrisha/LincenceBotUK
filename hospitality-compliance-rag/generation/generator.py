"""
generation/generator.py
────────────────────────
GPT-4o-mini integration for answer generation.

Mirrors the SME RAG architecture:
  - Lazy singleton OpenAI client
  - Low temperature (0.1) for grounded, factual answers
  - Citation extraction via regex
  - Fallback detection
  - Greeting short-circuit (no API call for greetings)
  - is_safety_query detection → injects safety addendum into prompt

SAFE for runtime import: depends only on openai, python-dotenv, retrieval outputs.
sentence-transformers is NOT imported here or anywhere in this import chain.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from generation.prompts import build_prompt, FALLBACK_PHRASE
from retrieval.rrf_fusion import RetrievedChunk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE     = float(os.getenv("OPENAI_TEMPERATURE", 0.1))
MAX_TOKENS      = int(os.getenv("OPENAI_MAX_TOKENS", 768))

# ---------------------------------------------------------------------------
# OpenAI client singleton
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to .env or Streamlit secrets."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Safety and greeting detection
# ---------------------------------------------------------------------------

_SAFETY_PATTERNS = re.compile(
    r"\b(revoc|review\s+hearing|court\s+appeal|police\s+objection|"
    r"criminal|arrest|prosecution|fine|closure\s+order|licence\s+strip|"
    r"underage|sell\s+to\s+minor|child\s+protection)\b",
    re.IGNORECASE,
)

_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|"
    r"what'?s\s+up|greetings)\b[!?.]*\s*$",
    re.IGNORECASE,
)

GREETING_RESPONSE = (
    "Hello! I'm the UK Hospitality Licensing Compliance Assistant. "
    "I can help you with questions about premises licences, personal licences, "
    "temporary event notices, licensing objectives, and other aspects of "
    "the Licensing Act 2003 (England and Wales). "
    "How can I help you today?"
)

# Citation regex — matches [Document Name, s.X] or [Source, para Y] style
_CITATION_RE = re.compile(r"\[[^\[\]]{5,120}\]")


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]
    citations: list[str]
    is_fallback: bool
    is_greeting: bool
    is_safety: bool
    latency_seconds: float
    model: str = OPENAI_MODEL
    retrieved_count: int = 0


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
) -> RAGResponse:
    """
    Generate an answer from retrieved context using GPT-4o-mini.

    Args:
        query:            Cleaned user query string.
        retrieved_chunks: Top-k chunks from rrf_fusion.hybrid_retrieve().

    Returns:
        RAGResponse with answer, citations, and diagnostic flags.
    """
    t0 = time.perf_counter()

    # --- Greeting short-circuit --------------------------------------------
    if _GREETING_PATTERNS.match(query):
        return RAGResponse(
            answer=GREETING_RESPONSE,
            sources=[],
            citations=[],
            is_fallback=False,
            is_greeting=True,
            is_safety=False,
            latency_seconds=time.perf_counter() - t0,
            retrieved_count=0,
        )

    is_safety = bool(_SAFETY_PATTERNS.search(query))

    # --- Build prompt -------------------------------------------------------
    system_prompt, user_message = build_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        is_safety_query=is_safety,
    )

    # --- Call OpenAI --------------------------------------------------------
    client = _get_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )

    answer = response.choices[0].message.content or ""
    latency = time.perf_counter() - t0

    # --- Post-processing ----------------------------------------------------
    is_fallback = FALLBACK_PHRASE.lower() in answer.lower()
    citations = _CITATION_RE.findall(answer)

    if not citations and not is_fallback:
        # Warn in output — the prompt instructs the model to always cite
        pass

    return RAGResponse(
        answer=answer,
        sources=retrieved_chunks,
        citations=citations,
        is_fallback=is_fallback,
        is_greeting=False,
        is_safety=is_safety,
        latency_seconds=round(latency, 3),
        model=OPENAI_MODEL,
        retrieved_count=len(retrieved_chunks),
    )
