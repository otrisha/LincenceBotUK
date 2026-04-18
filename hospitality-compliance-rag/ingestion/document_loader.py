"""
ingestion/document_loader.py
────────────────────────────
Two responsibilities:

1. DOWNLOAD — fetches the 7 UK licensing source documents (3 PDFs via
   requests streaming, 4 HTML pages scraped with BeautifulSoup) into
   data/raw/, then writes SHA-256 hashes to data/processed/document_hashes.json.

2. LOAD — reads every PDF in data/raw/ with pdfplumber and returns a list
   of LoadedDocument objects for the chunker to consume.

INGESTION ONLY — never imported by the Streamlit app or any runtime module.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pdfplumber
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
HASH_FILE = PROCESSED_DIR / "document_hashes.json"

_HEADERS = {
    "User-Agent": (
        "LicenceBotUK/1.0 (educational compliance tool; "
        "contact: orjitrisha@gmail.com)"
    )
}

# ---------------------------------------------------------------------------
# Document manifest
# ---------------------------------------------------------------------------

# Each entry: (filename, url, kind)
# kind = "pdf" | "html"
DOCUMENT_MANIFEST: list[tuple[str, str, str]] = [
    (
        "section_182_guidance.pdf",
        "https://assets.publishing.service.gov.uk/media/67b73b7b78dd6cacb71c6ac8/"
        "Revised_guidance_issued_under_section_182_of_the_Licensing_Act_2003_"
        "-_October+2024+_1_.pdf",
        "pdf",
    ),
    (
        "licensing_act_2003.pdf",
        "https://www.legislation.gov.uk/ukpga/2003/17/pdfs/ukpga_20030017_en.pdf",
        "pdf",
    ),
    (
        "lga_councillors_handbook.pdf",
        "https://www.local.gov.uk/sites/default/files/documents/"
        "10%2036_Licensing_Act_2003_V04%203_1.pdf",
        "pdf",
    ),
    (
        "ten_overview.txt",
        "https://www.gov.uk/find-licences/temporary-events-notice",
        "html",
    ),
    (
        "alcohol_licensing_hub.txt",
        "https://www.gov.uk/guidance/alcohol-licensing",
        "html",
    ),
    (
        "challenge_25_guidance.txt",
        "https://rasg.org.uk/",
        "html",
    ),
    (
        "personal_licence_guidance.txt",
        "https://www.gov.uk/personal-licence-to-sell-alcohol",
        "html",
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers — downloading
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_pdf(filename: str, url: str, dest_dir: Path) -> dict:
    """
    Download a PDF via streaming requests and save to dest_dir.

    Returns a metadata dict for the hash file, or raises on failure.
    """
    dest = dest_dir / filename
    print(f"  [{filename}] Downloading PDF ...")
    print(f"    URL: {url}")

    resp = requests.get(url, headers=_HEADERS, stream=True, timeout=60)
    resp.raise_for_status()

    bytes_written = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    digest = _sha256(dest)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"    Saved: {dest.name}  ({bytes_written:,} bytes)  SHA-256: {digest[:16]}...")

    return {
        "hash": digest,
        "url": url,
        "downloaded_at": ts,
        "file_size_bytes": bytes_written,
    }


_NOISE_PATTERNS = re.compile(
    r"(cookie|banner|breadcrumb|sidebar|skip-link|back-to-top|"
    r"related-navigation|gem-c-print-link|govuk-header|govuk-footer|"
    r"govuk-breadcrumbs|govuk-tabs__list)",
    re.IGNORECASE,
)


def _clean_html(soup: BeautifulSoup) -> str:
    """
    Extract main content from a BeautifulSoup tree as plain text.

    Priority: <main> -> <article> -> id="content" -> <body>
    Strips nav, header, footer, aside, cookie banners, scripts, styles.
    Converts h1/h2/h3 to ## / ### / #### Markdown headings.
    """
    # Remove noise elements by tag name first (safe — not dependent on tree position)
    for tag in list(soup.find_all(
        ["script", "style", "nav", "header", "footer", "aside",
         "noscript", "iframe", "form"]
    )):
        tag.decompose()

    # Remove common cookie-banner / skip-link patterns by class / id.
    # Snapshot with list() first; skip tags already removed from the tree
    # (decomposing a parent nullifies its children in the list).
    for tag in list(soup.find_all(True)):
        try:
            cls = " ".join(tag.get("class") or [])
            tag_id = tag.get("id") or ""
        except (AttributeError, TypeError):
            # Tag was already decomposed as a child of an earlier removal
            continue
        if _NOISE_PATTERNS.search(cls) or _NOISE_PATTERNS.search(tag_id):
            tag.decompose()

    # Locate the main content region
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find("body")
        or soup
    )

    lines: list[str] = []
    for element in main.descendants:
        if not hasattr(element, "name"):
            # NavigableString — only collect if parent is a text-bearing tag
            parent = element.parent
            if parent and parent.name in {
                "p", "li", "td", "th", "dt", "dd", "span", "div",
                "strong", "em", "b", "i", "a", "label",
            }:
                text = element.strip()
                if text:
                    lines.append(text)
        elif element.name == "h1":
            text = element.get_text(separator=" ", strip=True)
            if text:
                lines.append(f"\n## {text}")
        elif element.name == "h2":
            text = element.get_text(separator=" ", strip=True)
            if text:
                lines.append(f"\n### {text}")
        elif element.name == "h3":
            text = element.get_text(separator=" ", strip=True)
            if text:
                lines.append(f"\n#### {text}")
        elif element.name in {"p", "li", "dt", "dd"}:
            text = element.get_text(separator=" ", strip=True)
            if text:
                lines.append(text)

    # Collapse duplicate blank lines
    raw = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return cleaned


def _scrape_html(filename: str, url: str, dest_dir: Path) -> dict:
    """
    Scrape a GOV.UK / external HTML page and save cleaned text to dest_dir.

    Returns a metadata dict for the hash file, or raises on failure.
    """
    dest = dest_dir / filename
    print(f"  [{filename}] Scraping HTML ...")
    print(f"    URL: {url}")

    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    text = _clean_html(soup)

    with open(dest, "w", encoding="utf-8") as f:
        f.write(f"Source: {url}\n")
        f.write(f"Fetched: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        f.write("=" * 72 + "\n\n")
        f.write(text)

    bytes_written = dest.stat().st_size
    digest = _sha256(dest)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"    Saved: {dest.name}  ({bytes_written:,} bytes)  SHA-256: {digest[:16]}...")

    return {
        "hash": digest,
        "url": url,
        "downloaded_at": ts,
        "file_size_bytes": bytes_written,
    }


def _save_hashes(hashes: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)
    print(f"\n  Hash manifest written -> {HASH_FILE}")


# ---------------------------------------------------------------------------
# Public download API
# ---------------------------------------------------------------------------

def download_documents(
    dest_dir: Path = RAW_DIR,
    *,
    delay_seconds: float = 2.0,
) -> dict[str, dict]:
    """
    Download / scrape all 7 source documents into dest_dir.

    Continues on per-document failures (logs and skips).
    Returns the hash manifest dict and writes it to HASH_FILE.
    Prints a summary at the end.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing hashes so reruns only log — they still re-download to
    # refresh the file if a URL has changed.
    hashes: dict[str, dict] = {}
    if HASH_FILE.exists():
        with open(HASH_FILE, encoding="utf-8") as f:
            hashes = json.load(f)

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    total = len(DOCUMENT_MANIFEST)
    print(f"\nDownloading {total} documents to {dest_dir} ...\n")

    for i, (filename, url, kind) in enumerate(DOCUMENT_MANIFEST, start=1):
        print(f"[{i}/{total}] {filename}")
        try:
            if kind == "pdf":
                meta = _download_pdf(filename, url, dest_dir)
            else:
                meta = _scrape_html(filename, url, dest_dir)

            hashes[filename] = meta
            succeeded.append(filename)

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"    ERROR — {reason}")
            failed.append((filename, reason))

        # Polite delay between requests (skip after last item)
        if i < total:
            time.sleep(delay_seconds)

    _save_hashes(hashes)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  Downloaded: {len(succeeded)}/{total} documents successfully")
    if failed:
        print(f"  Failed    : {len(failed)}")
        for name, reason in failed:
            print(f"    FAIL {name}: {reason}")
    else:
        print("  Failed    : 0")
    print("=" * 60)

    return hashes


# ---------------------------------------------------------------------------
# Data structures (used by chunker)
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
    filename: str                  # e.g. "licensing_act_2003.pdf"
    stem: str                      # filename without extension
    blocks: list[DocumentBlock] = field(default_factory=list)
    total_pages: int = 0


# ---------------------------------------------------------------------------
# Internal helpers — PDF loading
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
    """Map relative font size to a heading level. Returns 0 for body text."""
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
    if m.group(1):
        return m.group(1).rstrip(".")
    return f"{m.group(2)} {m.group(3)}"


# ---------------------------------------------------------------------------
# Public load API
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
            "Run download_documents() first to fetch the source documents."
        )

    documents: list[LoadedDocument] = []
    for path in pdf_files:
        print(f"  Loading: {path.name} ...", end=" ", flush=True)
        doc = load_document(path)
        print(f"{doc.total_pages} pages, {len(doc.blocks)} blocks")
        documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_documents()
