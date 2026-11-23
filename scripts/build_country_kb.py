"""
Build the Country PDF knowledge base in ChromaDB.

This script reads all PDFs from the country_pdfs/ directory,
chunks them, generates embeddings, and stores them in a
separate ChromaDB collection (country_kb/).

Usage:
    python -m scripts.build_country_kb
    python -m scripts.build_country_kb --chunk-size 800 --chunk-overlap 150
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    COUNTRY_PDF_DIR,
    COUNTRY_KB_PATH,
    COUNTRY_KB_COLLECTION,
    PDF_CHUNK_SIZE,
    PDF_CHUNK_OVERLAP,
    PDF_COUNTRY_MAP,
)
from src.embeddings import EmbeddingModel
from src.pdf_chunker import process_all_pdfs
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build the country PDF knowledge base in ChromaDB."
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=str(COUNTRY_PDF_DIR),
        help=f"Directory containing country PDFs (default: {COUNTRY_PDF_DIR})",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=COUNTRY_KB_PATH,
        help=f"ChromaDB storage path (default: {COUNTRY_KB_PATH})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COUNTRY_KB_COLLECTION,
        help=f"ChromaDB collection name (default: {COUNTRY_KB_COLLECTION})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=PDF_CHUNK_SIZE,
        help=f"Characters per chunk (default: {PDF_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=PDF_CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {PDF_CHUNK_OVERLAP})",
    )

    args = parser.parse_args()

    # 1. Process PDFs into chunks
    logger.info(f"Processing PDFs from: {args.pdf_dir}")
    documents = process_all_pdfs(
        pdf_dir=args.pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        country_map=PDF_COUNTRY_MAP,
    )

    if not documents:
        logger.error("No documents produced from PDFs.")
        return

    logger.info(f"Total document chunks: {len(documents)}")

    # 2. Generate embeddings
    logger.info("Loading embedding model...")
    embedding_model = EmbeddingModel()

    logger.info("Generating embeddings...")
    embeddings = embedding_model.generate(documents)

    # 3. Store in ChromaDB
    logger.info(f"Connecting to ChromaDB at: {args.db_path}")
    vector_store = VectorStore(
        db_path=args.db_path,
        collection_name=args.collection,
    )

    logger.info("Adding documents to ChromaDB...")
    vector_store.add_documents(documents, embeddings)

    stored_count = vector_store.count()

    logger.info("Indexing complete!")
    logger.info(f"Documents expected: {len(documents)}")
    logger.info(f"Documents in ChromaDB: {stored_count}")

    assert stored_count == len(documents), (
        f"ChromaDB count mismatch! "
        f"Expected {len(documents)}, found {stored_count}"
    )

    logger.info("Country PDF knowledge base built successfully!")


if __name__ == "__main__":
    main()
