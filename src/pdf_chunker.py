"""
Extract text from PDF files, chunk it, and produce RAG documents.

Each PDF is split into overlapping text chunks that become
individual documents in the vector store.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import pymupdf

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text from a PDF file using PyMuPDF."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append(text)

    doc.close()

    full_text = "\n\n".join(pages)
    logger.info(
        f"Extracted {len(pages)} pages, "
        f"{len(full_text)} chars from {pdf_path.name}"
    )

    return full_text


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size."
        )

    if not text.strip():
        return []

    chunks = []
    text_length = len(text)
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to break at a sentence boundary in the second half
        # of the window, so chunks never shrink below half size.
        if end < text_length:
            for sep in [". ", ".\n", "?\n", "!\n", "\n\n"]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # The last window reached the end of the text: stop here,
        # otherwise the overlap step would emit a duplicate tail chunk.
        if end >= text_length:
            break

        # Step back by the overlap, but always move forward.
        start = max(end - chunk_overlap, start + 1)

    return chunks


def process_pdf(
    pdf_path: str | Path,
    source_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    country: str = "unknown",
) -> List[Dict[str, Any]]:
    """
    Process a single PDF into RAG documents.

    Args:
        pdf_path: Path to the PDF file.
        source_name: Unique identifier for this PDF (used in chunk IDs).
        country: Country the PDF describes (e.g., "india").
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of document dicts with 'text' and 'metadata' keys.
    """

    full_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(full_text, chunk_size, chunk_overlap)

    documents = []

    for i, chunk in enumerate(chunks):
        doc_id = f"{source_name}_chunk_{i:04d}"

        documents.append({
            "text": chunk,
            "metadata": {
                "data_id": doc_id,
                "source": source_name,
                "country": country,
                "source_file": Path(pdf_path).name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_type": "pdf",
            },
        })

    logger.info(
        f"Created {len(documents)} chunks from {Path(pdf_path).name}"
    )

    return documents


def process_all_pdfs(
    pdf_dir: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    country_map: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Process all PDF files in a directory into RAG documents.

    Args:
        pdf_dir: Directory containing PDF files.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        country_map: PDF filename -> country. Unlisted files get
            country "unknown".

    Returns:
        Combined list of all document dicts.
    """

    pdf_dir = Path(pdf_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    country_map = country_map or {}
    all_documents = []

    for pdf_file in pdf_files:
        # Derive a clean source name from the filename
        source_name = pdf_file.stem.lower().replace(" ", "_")

        country = country_map.get(pdf_file.name, "unknown")
        if country == "unknown":
            logger.warning(
                f"No country mapped for {pdf_file.name}; "
                "add it to PDF_COUNTRY_MAP in config.py"
            )

        documents = process_pdf(
            pdf_file,
            source_name=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            country=country,
        )

        all_documents.extend(documents)

    logger.info(
        f"Total: {len(all_documents)} chunks from "
        f"{len(pdf_files)} PDFs"
    )

    return all_documents
