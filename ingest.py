"""
Load PDFs from disk, extract text page-by-page, and chunk it.

Uses PyMuPDF (fitz) because it's fast and handles multi-column academic
layouts noticeably better than pypdf/pdfplumber for reading order.
"""
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from config import PDF_DIR, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    id: str
    text: str
    source: str          # filename
    page_start: int
    page_end: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    """Collapse hyphenation breaks and excess whitespace typical of PDF extraction."""
    text = re.sub(r"-\n(?=[a-z])", "", text)     # de-hyphenate line-wrapped words
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[str]:
    """Return list of per-page cleaned text."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        raw = page.get_text("text")
        pages.append(_clean_text(raw))
    doc.close()
    return pages


def _split_into_windows(pages: list[str], size: int, overlap: int) -> Iterator[tuple[str, int, int]]:
    """
    Concatenate pages with page markers, then slide a char window across the
    whole document so chunks aren't artificially cut at page boundaries.
    Yields (chunk_text, page_start, page_end).
    """
    # Build a single string with (char_offset -> page_number) breakpoints
    full_text = ""
    offsets = []  # (start_char, page_number)
    for i, page_text in enumerate(pages, start=1):
        offsets.append((len(full_text), i))
        full_text += page_text + "\n\n"

    def page_at(char_pos: int) -> int:
        page = offsets[0][1]
        for start, pg in offsets:
            if start <= char_pos:
                page = pg
            else:
                break
        return page

    start = 0
    n = len(full_text)
    if n == 0:
        return
    while start < n:
        end = min(start + size, n)
        # try not to cut mid-sentence: extend to next period/newline within 200 chars
        if end < n:
            lookahead = full_text[end:end + 200]
            m = re.search(r"[.\n]", lookahead)
            if m:
                end += m.end()
        chunk_text = full_text[start:end].strip()
        if chunk_text:
            yield chunk_text, page_at(start), page_at(max(end - 1, start))
        if end >= n:
            break
        start = end - overlap


def chunk_pdf(pdf_path: Path) -> list[Chunk]:
    pages = extract_pages(pdf_path)
    chunks = []
    for idx, (text, p_start, p_end) in enumerate(
        _split_into_windows(pages, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    ):
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=text,
                source=pdf_path.name,
                page_start=p_start,
                page_end=p_end,
                chunk_index=idx,
            )
        )
    return chunks


def load_and_chunk_all(pdf_dir: Path = PDF_DIR) -> list[Chunk]:
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Drop your research papers there first."
        )
    all_chunks = []
    for pdf_path in pdf_files:
        try:
            chunks = chunk_pdf(pdf_path)
            print(f"  {pdf_path.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠ failed on {pdf_path.name}: {e}")
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    print(f"\nTotal: {len(chunks)} chunks from PDFs in {PDF_DIR}")
    if chunks:
        print("\nSample chunk:")
        print(f"  source={chunks[0].source} pages={chunks[0].page_start}-{chunks[0].page_end}")
        print(f"  text preview: {chunks[0].text[:200]}...")
