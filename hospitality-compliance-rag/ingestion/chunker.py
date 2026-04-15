"""
ingestion/chunker.py
────────────────────
Domain-adaptive chunking for UK hospitality licensing documents.

Strategy selection is based on document name heuristics:
  - Legislation / Acts          → section_chunker  (structured by numbered sections)
  - Guidance / Circulars        → section_chunker  (structured by headings)
  - FAQ / Interview documents   → qa_pair_chunker  (Q&A pairs)
  - Enforcement / Case studies  → fault_block_chunker (scenario blocks)
  - All others                  → section_chunker  (safe default)

Chunk size limits mirror the SME RAG prototype:
  MAX_CHUNK_TOKENS = 600  (whitespace-token approximation)
  MIN_CHUNK_TOKENS = 80   (minimum before merging with predecessor)

INGESTION ONLY — never imported by the Streamlit app or any runtime module.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from ingestion.document_loader import DocumentBlock, LoadedDocument

# ---------------------------------------------------------------------------
# Config — read from env with sensible defaults
# ---------------------------------------------------------------------------

MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", 600))
MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", 80))

# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk ready for BM25 indexing and Pinecone upsert."""
    chunk_id: str                    # globally unique, URL-safe
    source_document: str             # PDF filename stem
    text: str                        # full text sent to the LLM
    bm25_text: str                   # text indexed by BM25 (may differ)
    heading: str                     # nearest heading above the chunk
    section_number: Optional[str]    # e.g. "3.2" or "SCHEDULE 1"
    page_number: int                 # page where the chunk begins (1-based)
    chunk_index: int                 # position within the document (0-based)
    token_count: int                 # approximate whitespace-token count
    topic_category: str = "general"  # classified topic
    chunking_method: str = "section" # which strategy produced this chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting (not BPE)."""
    return len(text.split())


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a lowercase URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len]


def _make_chunk_id(source: str, index: int, label: str) -> str:
    """Generate a deterministic chunk ID."""
    slug = _slugify(label)
    return f"{_slugify(source, 20)}-{index:04d}-{slug}"


def _extract_section_number(heading: str) -> Optional[str]:
    """Pull a leading section/paragraph number from a heading string."""
    m = re.match(
        r"^(\d+(?:\.\d+)*\.?)\s+"
        r"|^(PART|CHAPTER|SCHEDULE|SECTION|ANNEX)\s+([IVXLCDM\d]+)",
        heading.strip(),
        re.IGNORECASE,
    )
    if not m:
        return None
    if m.group(1):
        return m.group(1).rstrip(".")
    return f"{m.group(2).upper()} {m.group(3).upper()}"


_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "licensing_objectives":   ["licensing objective", "crime", "public safety", "public nuisance", "child protection"],
    "personal_licence":       ["personal licence", "dps", "designated premises supervisor", "personal license"],
    "premises_licence":       ["premises licence", "premises license", "licensable activity", "regulated entertainment"],
    "temporary_event":        ["ten", "temporary event notice", "temporary event", "counter notice"],
    "late_night_levy":        ["late night levy", "lnl"],
    "early_morning":          ["early morning restriction", "emro"],
    "review":                 ["review", "revoke", "suspension", "modify conditions"],
    "appeals":                ["appeal", "magistrates court", "crown court"],
    "enforcement":            ["enforcement", "police", "environmental health", "closure order"],
    "fees":                   ["fee", "fees", "charge", "payment", "cost"],
    "scotland":               ["scotland", "scottish"],
    "northern_ireland":       ["northern ireland"],
}


def _classify_topic(text: str) -> str:
    """Keyword-score the text against topic categories; return best match."""
    text_lower = text.lower()
    best_topic = "general"
    best_score = 0
    for topic, keywords in _TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def _merge_short(chunks: list[Chunk]) -> list[Chunk]:
    """
    Merge any chunk below MIN_CHUNK_TOKENS into its predecessor.
    This prevents degenerate single-sentence chunks.
    """
    if not chunks:
        return chunks
    merged: list[Chunk] = [chunks[0]]
    for chunk in chunks[1:]:
        if chunk.token_count < MIN_CHUNK_TOKENS and merged:
            prev = merged[-1]
            combined_text = prev.text + "\n\n" + chunk.text
            merged[-1] = Chunk(
                chunk_id=prev.chunk_id,
                source_document=prev.source_document,
                text=combined_text,
                bm25_text=prev.bm25_text + " " + chunk.bm25_text,
                heading=prev.heading,
                section_number=prev.section_number,
                page_number=prev.page_number,
                chunk_index=prev.chunk_index,
                token_count=_count_tokens(combined_text),
                topic_category=prev.topic_category,
                chunking_method=prev.chunking_method,
            )
        else:
            merged.append(chunk)
    # Re-index after merging
    for i, c in enumerate(merged):
        c.chunk_index = i
    return merged


# ---------------------------------------------------------------------------
# Strategy 1: section_chunker  (legislation, guidance, default)
# ---------------------------------------------------------------------------

def _group_by_heading(
    blocks: list[DocumentBlock],
    max_level: int = 2,
) -> list[tuple[str, Optional[str], int, list[DocumentBlock]]]:
    """
    Walk blocks and group them under the nearest heading at level ≤ max_level.
    Returns list of (heading_text, section_number, page_number, body_blocks).
    """
    groups: list[tuple[str, Optional[str], int, list[DocumentBlock]]] = []
    current_heading = "Introduction"
    current_section = None
    current_page = 1
    current_body: list[DocumentBlock] = []

    for block in blocks:
        if block.is_heading and block.heading_level <= max_level:
            if current_body:
                groups.append((current_heading, current_section, current_page, current_body))
            current_heading = block.text
            current_section = _extract_section_number(block.text)
            current_page = block.page_number
            current_body = []
        else:
            current_body.append(block)

    if current_body:
        groups.append((current_heading, current_section, current_page, current_body))

    return groups


def section_chunker(doc: LoadedDocument) -> list[Chunk]:
    """
    Group text by heading structure (level ≤ 2).
    Split oversized sections recursively at level-3 headings.
    """
    groups = _group_by_heading(doc.blocks, max_level=2)
    raw_chunks: list[tuple[str, Optional[str], int, str]] = []

    for heading, section_num, page, body_blocks in groups:
        body_text = "\n".join(b.text for b in body_blocks).strip()
        if not body_text:
            continue
        if _count_tokens(body_text) <= MAX_CHUNK_TOKENS:
            raw_chunks.append((heading, section_num, page, body_text))
        else:
            # Split at level-3 headings
            sub_heading = heading
            sub_section = section_num
            sub_page = page
            sub_lines: list[str] = []
            for block in body_blocks:
                if block.is_heading and block.heading_level == 3:
                    if sub_lines:
                        raw_chunks.append((sub_heading, sub_section, sub_page, "\n".join(sub_lines)))
                    sub_heading = block.text
                    sub_section = _extract_section_number(block.text) or section_num
                    sub_page = block.page_number
                    sub_lines = []
                else:
                    sub_lines.append(block.text)
            if sub_lines:
                raw_chunks.append((sub_heading, sub_section, sub_page, "\n".join(sub_lines)))

    chunks: list[Chunk] = []
    for i, (heading, section_num, page, body) in enumerate(raw_chunks):
        bm25_text = f"{heading} {body}"
        topic = _classify_topic(bm25_text)
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.stem, i, heading),
            source_document=doc.stem,
            text=f"{heading}\n\n{body}",
            bm25_text=bm25_text,
            heading=heading,
            section_number=section_num,
            page_number=page,
            chunk_index=i,
            token_count=_count_tokens(body),
            topic_category=topic,
            chunking_method="section",
        ))

    return _merge_short(chunks)


# ---------------------------------------------------------------------------
# Strategy 2: qa_pair_chunker  (FAQ / interview documents)
# ---------------------------------------------------------------------------

_QA_RE = re.compile(
    r"Q\s*[\n:]\s*(.+?)\s*[\n]A\s*[\n:]\s*(.+?)(?=\nQ\s*[\n:]|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def qa_pair_chunker(doc: LoadedDocument) -> list[Chunk]:
    """
    Extract Q&A pairs from the document.
    bm25_text = question only (so keyword search targets the question, not the answer).
    Falls back to section_chunker if no Q&A pairs are found.
    """
    full_text = "\n".join(b.text for b in doc.blocks)
    pairs = _QA_RE.findall(full_text)

    if not pairs:
        return section_chunker(doc)

    chunks: list[Chunk] = []
    for i, (question, answer) in enumerate(pairs):
        question = question.strip()
        answer = answer.strip()
        full = f"Q: {question}\nA: {answer}"
        topic = _classify_topic(question)
        # Find approximate page by scanning blocks for the question text
        page = 1
        for block in doc.blocks:
            if question[:30].lower() in block.text.lower():
                page = block.page_number
                break
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.stem, i, question),
            source_document=doc.stem,
            text=full,
            bm25_text=question,   # BM25 sees question only
            heading=f"Q: {question[:80]}",
            section_number=None,
            page_number=page,
            chunk_index=i,
            token_count=_count_tokens(full),
            topic_category=topic,
            chunking_method="qa_pair",
        ))

    return _merge_short(chunks)


# ---------------------------------------------------------------------------
# Strategy 3: fault_block_chunker  (enforcement / case study documents)
# Splits on CASE: or SCENARIO: or INCIDENT: delimiters
# ---------------------------------------------------------------------------

_CASE_RE = re.compile(r"(?=CASE\s*:|SCENARIO\s*:|INCIDENT\s*:)", re.IGNORECASE)


def fault_block_chunker(doc: LoadedDocument) -> list[Chunk]:
    """
    Split on CASE:/SCENARIO:/INCIDENT: delimiters, one block per chunk.
    Falls back to section_chunker if no delimiters found.
    """
    full_text = "\n".join(b.text for b in doc.blocks)
    raw_blocks = _CASE_RE.split(full_text)
    raw_blocks = [b.strip() for b in raw_blocks if b.strip()]

    if len(raw_blocks) <= 1:
        return section_chunker(doc)

    chunks: list[Chunk] = []
    for i, block_text in enumerate(raw_blocks):
        first_line = block_text.splitlines()[0] if block_text else "Case"
        topic = _classify_topic(block_text)
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.stem, i, first_line),
            source_document=doc.stem,
            text=block_text,
            bm25_text=block_text,
            heading=first_line[:120],
            section_number=None,
            page_number=1,
            chunk_index=i,
            token_count=_count_tokens(block_text),
            topic_category=topic,
            chunking_method="fault_block",
        ))

    return _merge_short(chunks)


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

_FAQ_HINTS = re.compile(r"faq|interview|question|q&a", re.IGNORECASE)
_CASE_HINTS = re.compile(r"enforcement|case.stud|scenario|incident", re.IGNORECASE)


def chunk_document(doc: LoadedDocument) -> list[Chunk]:
    """
    Select the appropriate chunking strategy based on the document name.
    """
    name = doc.filename.lower()
    if _FAQ_HINTS.search(name):
        return qa_pair_chunker(doc)
    if _CASE_HINTS.search(name):
        return fault_block_chunker(doc)
    return section_chunker(doc)


def chunk_all_documents(documents: list[LoadedDocument]) -> list[Chunk]:
    """
    Chunk every loaded document and return a flat list.
    Chunk indices are local to each document (reset per doc).
    chunk_id provides global uniqueness via the source_document prefix.
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        print(f"  Chunked: {doc.filename} → {len(doc_chunks)} chunks "
              f"(strategy: {doc_chunks[0].chunking_method if doc_chunks else 'none'})")
        all_chunks.extend(doc_chunks)
    return all_chunks
