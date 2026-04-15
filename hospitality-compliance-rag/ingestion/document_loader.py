"""
ingestion/document_loader.py
────────────────────────────
Loads PDF documents from data/raw/ using pdfplumber.

Each page is analysed for headings and body text by examining font-size
metadata. Results are returned as a list of DocumentBlock objects that the
chunker consumes.

INGESTION ONLY — never imported by the Streamlit app or any runtime module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentBlock:
    """A single logical block of text extracted from a PDF page."""
    text: str
    page_number: int               # 1-based
    is_heading: bool = False
    heading_level: int = 0         # 1 = top-level, 2 = sub, 3 = sub-sub; 0 = body
    font_size: float = 0.0
    source_document: str = ""      # PDF filename (stem)


@dataclass
class LoadedDocument:
    """All blocks extracted from one PDF file."""
    filename: str                  # e.g. "Licensing_Act_2003_Guidance.pdf"
    stem: str                      # filename without extension
    blocks: list[DocumentBlock] = field(default_factory=list)
    total_pages: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^(\d+(\.\d+)*\.?\s+.{3,}|"            # numbered: "3.2 Licensing objectives"
    r"PART\s+[IVXLCDM\d]+|"                  # "PART IV"
    r"CHAPTER\s+\d+|"                         # "CHAPTER 2"
    r"SCHEDULE\s+\d*|"                        # "SCHEDULE 1"
    r"SECTION\s+\d+|"                         # "SECTION 5"
    r"ANNEX\s+[A-Z\d]+)",                     # "ANNEX A"
    re.IGNORECASE,
)

_SECTION_NUMBER_RE = re.compile(
    r"^(\d+(?:\.\d+)*\.?)\s+"               # "3.2." or "12"
    r"|^(PART|CHAPTER|SCHEDULE|SECTION|ANNEX)\s+([IVXLCDM\d]+)",
    re.IGNORECASE,
)


def _infer_heading_level(font_size: float, median_size: float) -> int:
    """
    Map relative font size to a heading level.
    Returns 0 for body text.
    """
    ratio = font_size / median_size if median_size else 1.0
    if ratio >= 1.5:
        return 1
    if ratio >= 1.25:
        return 2
    if ratio >= 1.10:
        return 3
    return 0


def _median_font_size(page) -> float:
    """Return the median font size across all chars on a page."""
    sizes = [c["size"] for c in page.chars if c.get("size")]
    if not sizes:
        return 12.0
    sizes.sort()
    mid = len(sizes) // 2
    return float(sizes[mid])


def _extract_section_number(text: str) -> Optional[str]:
    """Pull a leading section number from a heading string, or None."""
    m = _SECTION_NUMBER_RE.match(text.strip())
    if not m:
        return None
    # group 1 = numeric "3.2.", group 2+3 = "PART IV"
    if m.group(1):
        return m.group(1).rstrip(".")
    return f"{m.group(2)} {m.group(3)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(pdf_path: Path) -> LoadedDocument:
    """
    Extract all text blocks from a single PDF file.

    Heuristic heading detection uses font size relative to the page median.
    Text lines matching _HEADING_RE are also promoted to headings even if
    font size does not differ (some PDFs use bold rather than size changes).
    """
    doc = LoadedDocument(
        filename=pdf_path.name,
        stem=pdf_path.stem,
    )

    with pdfplumber.open(pdf_path) as pdf:
        doc.total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            median_size = _median_font_size(page)

            # Group characters into lines using bounding-box y-coordinate
            raw_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if not raw_text:
                continue

            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue

                # Approximate font size for the line by sampling chars
                line_chars = [
                    c for c in page.chars
                    if c.get("text", "").strip() and c.get("text") in line[:50]
                ]
                avg_size = (
                    sum(c["size"] for c in line_chars) / len(line_chars)
                    if line_chars else median_size
                )

                level = _infer_heading_level(avg_size, median_size)
                # Override: regex-matched headings are at least level 2
                if level == 0 and _HEADING_RE.match(line):
                    level = 2

                block = DocumentBlock(
                    text=line,
                    page_number=page_num,
                    is_heading=(level > 0),
                    heading_level=level,
                    font_size=round(avg_size, 1),
                    source_document=doc.stem,
                )
                doc.blocks.append(block)

    return doc


def load_all_documents(raw_dir: Path) -> list[LoadedDocument]:
    """
    Load every PDF in raw_dir.

    Returns documents sorted by filename for deterministic ordering.
    Skips non-PDF files silently.
    """
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {raw_dir}. "
            "Place your UK licensing documents there before running ingestion."
        )

    documents: list[LoadedDocument] = []
    for path in pdf_files:
        print(f"  Loading: {path.name} ...", end=" ", flush=True)
        doc = load_document(path)
        print(f"{doc.total_pages} pages, {len(doc.blocks)} blocks")
        documents.append(doc)

    return documents
