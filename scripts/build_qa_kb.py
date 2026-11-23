"""
Build the QA-pair ChromaDB knowledge base from processed training data.

This indexes the QA pairs from qa_documents/ into qa_kb/.
This was used for initial pipeline validation. The primary KB
is now built from Country PDFs via build_country_kb.py.

Usage:
    python -m scripts.build_qa_kb
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    QA_DOCUMENTS_DIR,
    QA_KB_PATH,
    QA_KB_COLLECTION,
)
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_FILES = [
    "bangladesh_train.json",
    "india_train.json",
    "qatar_train.json",
]


def load_processed_documents(file_path):
    """Load processed RAG documents from a JSON file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError(
            f"Expected a list of documents in {file_path}"
        )

    return documents


def main():
    """Build the QA-pair ChromaDB knowledge base."""

    all_documents = []

    for file_name in DATA_FILES:
        file_path = QA_DOCUMENTS_DIR / file_name
        documents = load_processed_documents(file_path)
        logger.info(
            f"Loaded {len(documents)} documents from {file_name}"
        )
        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No documents found to index.")

    logger.info(f"Total documents loaded: {len(all_documents)}")

    # Deduplicate by source + data_id
    seen_ids = set()
    unique_documents = []

    for document in all_documents:
        doc_id = (
            f"{document['metadata']['source']}_"
            f"{document['metadata']['data_id']}"
        )
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_documents.append(document)

    duplicates_removed = len(all_documents) - len(unique_documents)
    if duplicates_removed > 0:
        logger.info(
            f"Removed {duplicates_removed} duplicate documents."
        )

    all_documents = unique_documents
    logger.info(f"Unique documents to index: {len(all_documents)}")

    logger.info("Loading embedding model...")
    embedding_model = EmbeddingModel()

    logger.info("Generating embeddings...")
    embeddings = embedding_model.generate(all_documents)

    logger.info("Connecting to ChromaDB...")
    vector_store = VectorStore(
        db_path=QA_KB_PATH,
        collection_name=QA_KB_COLLECTION,
    )

    logger.info("Adding documents to ChromaDB...")
    vector_store.add_documents(all_documents, embeddings)

    stored_count = vector_store.count()

    logger.info("Indexing complete!")
    logger.info(f"Documents expected: {len(all_documents)}")
    logger.info(f"Documents in ChromaDB: {stored_count}")

    assert stored_count == len(all_documents), (
        f"ChromaDB count mismatch! "
        f"Expected {len(all_documents)}, found {stored_count}"
    )

    logger.info("QA-pair knowledge base built successfully!")


if __name__ == "__main__":
    main()
