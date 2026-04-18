"""
ingestion/chunker.py
────────────────────
Domain-adaptive chunking for UK hospitality licensing documents.

Handles both PDF and TXT source files from data/raw/.

Strategy selection is based on document name heuristics:
  - Legislation / Acts          -> section_chunker  (structured by numbered sections)
  - Guidance / Circulars        -> section_chunker  (structured by headings)
  - FAQ / Interview documents   -> qa_pair_chunker  (Q&A pairs)
  - Enforcement / Case studies  -> fault_block_chunker (scenario blocks)
  - All others                  -> section_chunker  (safe default)

Chunk size limits mirror the SME RAG prototype:
  MAX_CHUNK_TOKENS = 600  (whitespace-token approximation)
  MIN_CHUNK_TOKENS = 80   (minimum before merging with predecessor)

INGESTION ONLY — never imported by the Streamlit app or any runtime module.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pypdf

from ingestion.document_loader import DocumentBlock, LoadedDocument

# ---------------------------------------------------------------------------
# Config — read from env with sensible defaults
# ---------------------------------------------------------------------------

MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", 600))
MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", 80))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
RAW_DIR       = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
CHUNKS_JSON   = PROCESSED_DIR / "chunks.json"

# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk ready for BM25 indexing and Pinecone upsert."""
    chunk_id:        str            # globally unique, URL-safe
    source_document: str            # filename with extension, e.g. "section_182_guidance.pdf"
    text:            str            # full text sent to the LLM
    bm25_text:       str            # text indexed by BM25 (may differ from text)
    heading:         str            # nearest heading above the chunk
    section_number:  Optional[str]  # e.g. "3.2" or "SCHEDULE 1"; None if absent
    page_number:     Optional[int]  # 1-based page (None for TXT files)
    chunk_index:     int            # position within the document (0-based)
    token_count:     int            # approximate whitespace-token count
    char_count:      int            # character length of text
    topic_category:  str = "general"   # classified topic
    chunking_method: str = "section"   # which strategy produced this chunk
    sub_chunk_index: Optional[int] = None  # set only when a chunk is split by the hard-char-limit pass


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
    """Generate a deterministic chunk ID: {source_slug}-{index:04d}-{label_slug}."""
    return f"{_slugify(source, 20)}-{index:04d}-{_slugify(label)}"


def _extract_section_number(heading: str) -> Optional[str]:
    """Pull a leading section/paragraph number from a heading string, or None."""
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
    "licensing_objectives": ["licensing objective", "crime", "public safety",
                              "public nuisance", "child protection"],
    "personal_licence":     ["personal licence", "dps", "designated premises supervisor",
                              "personal license"],
    "premises_licence":     ["premises licence", "premises license", "licensable activity",
                              "regulated entertainment"],
    "temporary_event":      ["ten", "temporary event notice", "temporary event",
                              "counter notice"],
    "late_night_levy":      ["late night levy", "lnl"],
    "early_morning":        ["early morning restriction", "emro"],
    "review":               ["review", "revoke", "suspension", "modify conditions"],
    "appeals":              ["appeal", "magistrates court", "crown court"],
    "enforcement":          ["enforcement", "police", "environmental health",
                              "closure order"],
    "fees":                 ["fee", "fees", "charge", "payment", "cost"],
    "scotland":             ["scotland", "scottish"],
    "northern_ireland":     ["northern ireland"],
}


def _classify_topic(text: str) -> str:
    """Keyword-score text against topic categories; return the best match."""
    text_lower = text.lower()
    best_topic, best_score = "general", 0
    for topic, keywords in _TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score, best_topic = score, topic
    return best_topic


def _merge_short(chunks: list[Chunk]) -> list[Chunk]:
    """
    Merge any chunk below MIN_CHUNK_TOKENS into its predecessor.
    Prevents degenerate single-sentence chunks. Re-indexes after merging.
    """
    if not chunks:
        return chunks

    merged: list[Chunk] = [chunks[0]]
    for chunk in chunks[1:]:
        if chunk.token_count < MIN_CHUNK_TOKENS and merged:
            prev = merged[-1]
            combined = prev.text + "\n\n" + chunk.text
            merged[-1] = Chunk(
                chunk_id=prev.chunk_id,
                source_document=prev.source_document,
                text=combined,
                bm25_text=prev.bm25_text + " " + chunk.bm25_text,
                heading=prev.heading,
                section_number=prev.section_number,
                page_number=prev.page_number,
                chunk_index=prev.chunk_index,
                token_count=_count_tokens(combined),
                char_count=len(combined),
                topic_category=prev.topic_category,
                chunking_method=prev.chunking_method,
                sub_chunk_index=None,
            )
        else:
            merged.append(chunk)

    for i, c in enumerate(merged):
        c.chunk_index = i
    return merged


# ---------------------------------------------------------------------------
# PDF loader (pypdf — simple page-by-page, no font metadata)
# ---------------------------------------------------------------------------

# Ordered list of (compiled pattern, heading level) for line classification.
# More specific / higher-level patterns come first.
_PDF_HEADING_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"^(PART|CHAPTER)\s+[IVXLCDM\d]+", re.IGNORECASE),  1),
    (re.compile(r"^(SCHEDULE|ANNEX)\s+[A-Z\d]+",    re.IGNORECASE),  1),
    (re.compile(r"^\d{1,3}\.\d{1,3}\s+\S"),                          2),  # "1.2 Sub-section"
    (re.compile(r"^\d{1,3}\s+[A-Z][a-z]"),                           2),  # "1 Licensing objectives"
    (re.compile(r"^SECTION\s+\d+",                  re.IGNORECASE),  2),
]


def _detect_pdf_heading_level(line: str) -> int:
    """
    Classify a line extracted by pypdf as heading level 1, 2, 3, or body (0).

    Since pypdf yields no font-size metadata, classification is pattern-only.
    Short ALL-CAPS lines (3-80 chars) are also treated as level-2 headings.
    Lines longer than 150 chars are always body text.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 150:
        return 0
    for pattern, level in _PDF_HEADING_PATTERNS:
        if pattern.match(stripped):
            return level
    # Short ALL-CAPS line: e.g. "INTRODUCTION", "DEFINITIONS", "OVERVIEW"
    if stripped.isupper() and 3 <= len(stripped) <= 80:
        return 2
    return 0


def _load_pdf_pypdf(path: Path) -> LoadedDocument:
    """
    Load a PDF file using pypdf, extracting text page by page.

    Returns a LoadedDocument with DocumentBlock objects whose headings are
    inferred from text patterns (not font metrics, which pypdf does not expose).
    Compatible with all three chunking strategies.
    """
    doc = LoadedDocument(filename=path.name, stem=path.stem)
    reader = pypdf.PdfReader(str(path))
    doc.total_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            level = _detect_pdf_heading_level(line)
            doc.blocks.append(DocumentBlock(
                text=line,
                page_number=page_num,
                is_heading=(level > 0),
                heading_level=level,
                font_size=0.0,          # not available via pypdf
                source_document=doc.stem,
            ))

    return doc



# ---------------------------------------------------------------------------
# TXT loader (for scraped HTML pages saved as plain text)
# ---------------------------------------------------------------------------

# The HTML scraper in document_loader.py writes headings as:
#   ## H1 text       -> heading_level 1
#   ### H2 text      -> heading_level 2
#   #### H3 text     -> heading_level 3
# The first 3 lines of every .txt file are a metadata header (Source/Fetched/===)
# which we skip.

_TXT_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)")


def _load_txt_file(path: Path) -> LoadedDocument:
    """
    Load a scraped-HTML text file from data/raw/.

    Parses ## / ### / #### heading markers into DocumentBlock objects with
    correct heading levels. Page numbers are unavailable; page_number=0 is
    used as a sentinel (chunkers will convert 0 -> None in Chunk.page_number).
    """
    doc = LoadedDocument(filename=path.name, stem=path.stem)
    doc.total_pages = 0   # not applicable for text files

    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # Skip the 3-line file header written by the downloader
    # (Source: ..., Fetched: ..., ===...  plus any immediately following blank lines)
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("==="):
            start = i + 1
            break

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue

        m = _TXT_HEADING_RE.match(stripped)
        if m:
            hashes, heading_text = m.group(1), m.group(2).strip()
            level = len(hashes)          # ## -> 1, ### -> 2, #### -> 3
            doc.blocks.append(DocumentBlock(
                text=heading_text,
                page_number=0,           # sentinel: no page number for TXT
                is_heading=True,
                heading_level=level,
                font_size=0.0,
                source_document=doc.stem,
            ))
        else:
            doc.blocks.append(DocumentBlock(
                text=stripped,
                page_number=0,
                is_heading=False,
                heading_level=0,
                font_size=0.0,
                source_document=doc.stem,
            ))

    return doc


# ---------------------------------------------------------------------------
# Document loader dispatcher
# ---------------------------------------------------------------------------

def load_raw_documents(raw_dir: Path = RAW_DIR) -> list[LoadedDocument]:
    """
    Load all PDF and TXT files from raw_dir using the appropriate loader.

    PDFs  -> _load_pdf_pypdf  (page-by-page, pattern heading detection)
    TXTs  -> _load_txt_file   (heading-marker parsing)

    Returns documents sorted by filename for deterministic ordering.
    """
    paths = sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in {".pdf", ".txt"}
    )
    if not paths:
        raise FileNotFoundError(
            f"No PDF or TXT files found in {raw_dir}. "
            "Run ingestion/document_loader.py first to download source documents."
        )

    documents: list[LoadedDocument] = []
    for path in paths:
        suffix = path.suffix.lower()
        loader_name = "pypdf" if suffix == ".pdf" else "txt"
        print(f"  Loading [{loader_name}]: {path.name} ...", end=" ", flush=True)

        if suffix == ".pdf":
            doc = _load_pdf_pypdf(path)
        else:
            doc = _load_txt_file(path)

        print(f"{doc.total_pages or '-'} pages, {len(doc.blocks)} blocks")
        documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# Strategy 1: section_chunker  (legislation, guidance, default)
# ---------------------------------------------------------------------------

def _group_by_heading(
    blocks: list[DocumentBlock],
    max_level: int = 2,
) -> list[tuple[str, Optional[str], Optional[int], list[DocumentBlock]]]:
    """
    Walk blocks and group them under the nearest heading at level <= max_level.

    Returns list of (heading_text, section_number, page_number, body_blocks).
    page_number is None when all blocks carry the sentinel value 0.
    """
    groups: list[tuple[str, Optional[str], Optional[int], list[DocumentBlock]]] = []
    current_heading = "Introduction"
    current_section: Optional[str] = None
    current_page: Optional[int] = None
    current_body: list[DocumentBlock] = []

    for block in blocks:
        if block.is_heading and block.heading_level <= max_level:
            if current_body:
                groups.append((current_heading, current_section, current_page, current_body))
            current_heading = block.text
            current_section = _extract_section_number(block.text)
            current_page = block.page_number or None   # 0 -> None
            current_body = []
        else:
            if current_page is None and block.page_number:
                current_page = block.page_number
            current_body.append(block)

    if current_body:
        groups.append((current_heading, current_section, current_page, current_body))

    return groups


def section_chunker(doc: LoadedDocument) -> list[Chunk]:
    """
    Group text by heading structure (level <= 2).
    Split oversized sections recursively at level-3 headings.
    BM25 text = heading + body (heading prepended for keyword match boost).
    """
    groups = _group_by_heading(doc.blocks, max_level=2)
    raw_chunks: list[tuple[str, Optional[str], Optional[int], str]] = []

    for heading, section_num, page, body_blocks in groups:
        body_text = "\n".join(b.text for b in body_blocks).strip()
        if not body_text:
            continue

        if _count_tokens(body_text) <= MAX_CHUNK_TOKENS:
            raw_chunks.append((heading, section_num, page, body_text))
        else:
            # Oversized: split at level-3 headings within this section
            sub_heading = heading
            sub_section = section_num
            sub_page = page
            sub_lines: list[str] = []

            for block in body_blocks:
                if block.is_heading and block.heading_level == 3:
                    if sub_lines:
                        raw_chunks.append((sub_heading, sub_section, sub_page,
                                           "\n".join(sub_lines)))
                    sub_heading = block.text
                    sub_section = _extract_section_number(block.text) or section_num
                    sub_page = block.page_number or None
                    sub_lines = []
                else:
                    sub_lines.append(block.text)
            if sub_lines:
                raw_chunks.append((sub_heading, sub_section, sub_page,
                                   "\n".join(sub_lines)))

    chunks: list[Chunk] = []
    for i, (heading, section_num, page, body) in enumerate(raw_chunks):
        full_text = f"{heading}\n\n{body}"
        bm25_text = f"{heading} {body}"
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.filename, i, heading),
            source_document=doc.filename,
            text=full_text,
            bm25_text=bm25_text,
            heading=heading,
            section_number=section_num,
            page_number=page,
            chunk_index=i,
            token_count=_count_tokens(body),
            char_count=len(full_text),
            topic_category=_classify_topic(bm25_text),
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
    bm25_text = question only — keyword search targets the question, not the answer.
    Falls back to section_chunker if no Q&A pairs are found.
    """
    full_text = "\n".join(b.text for b in doc.blocks)
    pairs = _QA_RE.findall(full_text)

    if not pairs:
        return section_chunker(doc)

    chunks: list[Chunk] = []
    for i, (question, answer) in enumerate(pairs):
        question = question.strip()
        answer   = answer.strip()
        full     = f"Q: {question}\nA: {answer}"
        # Locate approximate page from blocks
        page: Optional[int] = None
        for block in doc.blocks:
            if question[:30].lower() in block.text.lower():
                page = block.page_number or None
                break
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.filename, i, question),
            source_document=doc.filename,
            text=full,
            bm25_text=question,      # BM25 indexes question only
            heading=f"Q: {question[:80]}",
            section_number=None,
            page_number=page,
            chunk_index=i,
            token_count=_count_tokens(full),
            char_count=len(full),
            topic_category=_classify_topic(question),
            chunking_method="qa_pair",
        ))

    return _merge_short(chunks)


# ---------------------------------------------------------------------------
# Strategy 3: fault_block_chunker  (enforcement / case study documents)
# ---------------------------------------------------------------------------

_CASE_SPLIT_RE = re.compile(
    r"(?=CASE\s*:|SCENARIO\s*:|INCIDENT\s*:)", re.IGNORECASE
)


def fault_block_chunker(doc: LoadedDocument) -> list[Chunk]:
    """
    Split on CASE: / SCENARIO: / INCIDENT: delimiters, one block per chunk.
    Falls back to section_chunker if no delimiters are found.
    """
    full_text = "\n".join(b.text for b in doc.blocks)
    raw_blocks = [b.strip() for b in _CASE_SPLIT_RE.split(full_text) if b.strip()]

    if len(raw_blocks) <= 1:
        return section_chunker(doc)

    chunks: list[Chunk] = []
    for i, block_text in enumerate(raw_blocks):
        first_line = block_text.splitlines()[0] if block_text else "Case"
        chunks.append(Chunk(
            chunk_id=_make_chunk_id(doc.filename, i, first_line),
            source_document=doc.filename,
            text=block_text,
            bm25_text=block_text,
            heading=first_line[:120],
            section_number=None,
            page_number=None,        # case blocks are not page-tied
            chunk_index=i,
            token_count=_count_tokens(block_text),
            char_count=len(block_text),
            topic_category=_classify_topic(block_text),
            chunking_method="fault_block",
        ))

    return _merge_short(chunks)


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

_FAQ_HINTS  = re.compile(r"faq|interview|question|q&a",          re.IGNORECASE)
_CASE_HINTS = re.compile(r"enforcement|case.stud|scenario|incident", re.IGNORECASE)


def chunk_document(doc: LoadedDocument) -> list[Chunk]:
    """Select and run the appropriate chunking strategy based on filename."""
    name = doc.filename.lower()
    if _FAQ_HINTS.search(name):
        return qa_pair_chunker(doc)
    if _CASE_HINTS.search(name):
        return fault_block_chunker(doc)
    return section_chunker(doc)


def chunk_all_documents(documents: list[LoadedDocument]) -> list[Chunk]:
    """
    Chunk every loaded document and return a flat combined list.
    chunk_index is local to each document; chunk_id is globally unique.
    Also used by run_ingestion.py (which supplies its own LoadedDocument list).
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc)
        before = len(doc_chunks)
        doc_chunks = split_oversized_chunks(doc_chunks)
        after = len(doc_chunks)
        strategy = doc_chunks[0].chunking_method if doc_chunks else "none"
        split_note = f" +{after - before} from char-limit splits" if after > before else ""
        print(f"  Chunked: {doc.filename} -> {after} chunks (strategy: {strategy}{split_note})")
        all_chunks.extend(doc_chunks)
    return all_chunks



# ---------------------------------------------------------------------------
# Hard character-limit pass  (runs after all strategy chunking is done)
# ---------------------------------------------------------------------------

_CHAR_LIMIT:   int = 1800
_CHAR_OVERLAP: int = 200


def _split_chunk(
    chunk: Chunk,
    max_chars: int = _CHAR_LIMIT,
    overlap: int = _CHAR_OVERLAP,
) -> list[Chunk]:
    """
    Split one oversized Chunk into sub-chunks of at most max_chars characters,
    with `overlap` characters of shared context between consecutive sub-chunks.

    Break preference (in order):
      1. Sentence boundary — last ". " before the max_chars cut point
      2. Word boundary     — last " " before the cut point
      3. Hard cut          — at max_chars if no whitespace is found

    Every sub-chunk inherits all metadata from the parent; sub_chunk_index
    is set to 0, 1, 2, ... to distinguish siblings.
    """
    text = chunk.text
    if len(text) <= max_chars:
        return [chunk]

    sub_chunks: list[Chunk] = []
    start = 0
    sub_idx = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            # Prefer sentence boundary: last ". " in (start+overlap .. end)
            # The lower bound (start+overlap) ensures we always make progress.
            lo = start + overlap
            bp = text.rfind(". ", lo, end)
            if bp != -1:
                end = bp + 2            # include the ". "
            else:
                # Fall back to last word boundary
                sp = text.rfind(" ", lo, end)
                if sp != -1:
                    end = sp + 1
                # else: hard cut at max_chars — no whitespace found

        segment = text[start:end].strip()
        if segment:
            sub_chunks.append(Chunk(
                chunk_id        = f"{chunk.chunk_id}-{sub_idx:02d}",
                source_document = chunk.source_document,
                text            = segment,
                bm25_text       = segment,
                heading         = chunk.heading,
                section_number  = chunk.section_number,
                page_number     = chunk.page_number,
                chunk_index     = chunk.chunk_index,
                token_count     = _count_tokens(segment),
                char_count      = len(segment),
                topic_category  = chunk.topic_category,
                chunking_method = chunk.chunking_method,
                sub_chunk_index = sub_idx,
            ))
            sub_idx += 1

        if end >= len(text):
            break

        # Next sub-chunk starts `overlap` chars before where this one ended,
        # but never further back than `start + 1` (guarantees forward progress).
        start = max(end - overlap, start + 1)

    return sub_chunks


def split_oversized_chunks(
    chunks: list[Chunk],
    max_chars: int = _CHAR_LIMIT,
    overlap: int = _CHAR_OVERLAP,
) -> list[Chunk]:
    """
    Second-pass over a chunk list: any chunk exceeding max_chars is replaced
    by the sub-chunks produced by _split_chunk().  Chunks already within the
    limit are passed through unchanged (sub_chunk_index remains None).
    """
    result: list[Chunk] = []
    for chunk in chunks:
        if chunk.char_count > max_chars:
            result.extend(_split_chunk(chunk, max_chars, overlap))
        else:
            result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_chunks_json(chunks: list[Chunk], out_path: Path = CHUNKS_JSON) -> None:
    """
    Serialise chunks to data/processed/chunks.json.

    Output format per chunk:
      {
        "text": "...",
        "metadata": {
          "source_document": "section_182_guidance.pdf",
          "page_number": 3,          # null for TXT files
          "section_number": "1.2",   # null if absent
          "chunk_index": 0,
          "char_count": 412
        }
      }
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "text": c.text,
            "metadata": {
                "source_document": c.source_document,
                "page_number":     c.page_number,
                "section_number":  c.section_number,
                "chunk_index":     c.chunk_index,
                "sub_chunk_index": c.sub_chunk_index,
                "char_count":      c.char_count,
            },
        }
        for c in chunks
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n  Chunks saved -> {out_path}  ({len(records)} chunks)")


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def run_quality_checks(chunks: list[Chunk]) -> None:
    """
    Print a quality report to stdout.

    Reports:
      - Total chunk count
      - Average / min / max char_count
      - Chunks per source document
      - Chunks flagged as too small (< 50 chars) or too large (> 1800 chars)
    """
    if not chunks:
        print("  [WARN] No chunks produced — nothing to check.")
        return

    char_counts = [c.char_count for c in chunks]
    avg_chars   = sum(char_counts) / len(char_counts)
    min_chars   = min(char_counts)
    max_chars   = max(char_counts)

    # Per-document breakdown
    from collections import Counter
    per_doc: Counter = Counter(c.source_document for c in chunks)

    # Flag anomalies
    too_small = [c for c in chunks if c.char_count < 50]
    too_large = [c for c in chunks if c.char_count > 1800]

    print("\n" + "=" * 60)
    print("  CHUNK QUALITY REPORT")
    print("=" * 60)
    print(f"  Total chunks      : {len(chunks)}")
    print(f"  Avg char count    : {avg_chars:.0f}")
    print(f"  Min char count    : {min_chars}  (chunk_id: {min(chunks, key=lambda c: c.char_count).chunk_id})")
    print(f"  Max char count    : {max_chars}  (chunk_id: {max(chunks, key=lambda c: c.char_count).chunk_id})")

    print("\n  Chunks per document:")
    for doc_name, count in sorted(per_doc.items()):
        print(f"    {doc_name:<45} {count:>4} chunks")

    if too_small:
        print(f"\n  [FLAG] {len(too_small)} chunk(s) under 50 chars (potentially too small):")
        for c in too_small[:5]:
            preview = repr(c.text[:60])
            print(f"    chunk_id={c.chunk_id}  text={preview}")
        if len(too_small) > 5:
            print(f"    ... and {len(too_small) - 5} more")
    else:
        print("\n  [OK] No chunks under 50 chars")

    if too_large:
        print(f"\n  [FLAG] {len(too_large)} chunk(s) over 1800 chars (potentially too large):")
        for c in too_large[:5]:
            print(f"    chunk_id={c.chunk_id}  char_count={c.char_count}")
        if len(too_large) > 5:
            print(f"    ... and {len(too_large) - 5} more")
    else:
        print("  [OK] No chunks over 1800 chars")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Standalone runner:  python -m ingestion.chunker

    Loads all PDFs and TXT files from data/raw/, chunks them, runs quality
    checks, then saves to data/processed/chunks.json.
    """
    print("\n" + "=" * 60)
    print("  UK Hospitality Compliance RAG -- Chunker")
    print("=" * 60)

    print(f"\n[1/3] Loading documents from {RAW_DIR} ...")
    documents = load_raw_documents(RAW_DIR)
    print(f"  Total documents loaded: {len(documents)}")

    print("\n[2/3] Chunking documents ...")
    chunks = chunk_all_documents(documents)
    print(f"  Total chunks produced : {len(chunks)}")

    print("\n[3/3] Running quality checks ...")
    run_quality_checks(chunks)

    save_chunks_json(chunks, CHUNKS_JSON)

    # Show first 3 chunks from section_182_guidance.pdf
    print("\n" + "=" * 60)
    print("  SAMPLE: first 3 chunks from section_182_guidance.pdf")
    print("=" * 60)
    sample = [c for c in chunks if "section_182_guidance" in c.source_document][:3]
    for n, c in enumerate(sample, start=1):
        print(f"\n  --- Chunk {n} ---")
        print(f"  chunk_id       : {c.chunk_id}")
        print(f"  source_document: {c.source_document}")
        print(f"  page_number    : {c.page_number}")
        print(f"  section_number : {c.section_number}")
        print(f"  chunk_index    : {c.chunk_index}")
        print(f"  char_count     : {c.char_count}")
        print(f"  token_count    : {c.token_count}")
        print(f"  topic_category : {c.topic_category}")
        print(f"  chunking_method: {c.chunking_method}")
        print(f"  heading        : {c.heading}")
        preview = c.text[:300].replace("\n", " ")
        print(f"  text (first 300): {preview!r}")


if __name__ == "__main__":
    main()


