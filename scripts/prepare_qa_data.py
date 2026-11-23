"""
Rebuild qa_documents/ from the raw datasets in data/.

Loads and validates data/<country>/{train,dev,test}.json, then turns
each training record into one RAG document (input query, question and
answer) in qa_documents/<country>_train.json. Those files feed the
QA-pair knowledge base built by scripts/build_qa_kb.py.

Only the train split is written: dev and test must stay out of the
knowledge base so evaluation measures unseen questions.

Usage:
    python -m scripts.prepare_qa_data
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, QA_DOCUMENTS_DIR
from src.qa_data_loader import load_all_data
from src.qa_document_processor import process_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Loading datasets from {DATA_DIR}")
    all_data = load_all_data(DATA_DIR)

    for country, splits in all_data.items():
        output_path = QA_DOCUMENTS_DIR / f"{country}_train.json"
        documents = process_documents(splits["train"], output_path)
        logger.info(
            f"{country}: wrote {len(documents)} documents "
            f"to {output_path}"
        )

    logger.info("Processed data rebuilt successfully!")


if __name__ == "__main__":
    main()
