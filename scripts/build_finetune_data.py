"""
Build the chat-format fine-tuning data (train + dev) for LoRA training.

Loads the train and dev splits from data/{country}/, retrieves context
for each question from the chosen KB (--kb qa, the default, or
country_pdf), and writes finetune_data/training_data.jsonl. The test
split is never used. The prompt format is the same one
src/generator.py uses at answer time.

With the QA-pair KB, a training question's own record is in the KB
(it was indexed from the train split). Retrieving it would put the
ground-truth answer in the prompt, so the record itself is excluded
from its own retrieved context. Dev records are not in the KB.

Next step: train the adapter with scripts/train_lora.py (CUDA GPU).
Turning the adapter into the Ollama model (merge, GGUF, ollama create)
is documented in the README, not done by this script.

Usage:
    python -m scripts.build_finetune_data
    python -m scripts.build_finetune_data --country india --sample-size 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR,
    SUPPORTED_COUNTRIES,
    QA_KB_PATH,
    QA_KB_COLLECTION,
    COUNTRY_KB_PATH,
    COUNTRY_KB_COLLECTION,
    FINETUNE_OUTPUT_DIR,
    DEFAULT_TOP_K,
)
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import build_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


TRAINING_DATA_FILE = FINETUNE_OUTPUT_DIR / "training_data.jsonl"


def load_split(country: str, split: str) -> List[Dict[str, Any]]:
    """Load a specific data split for a country."""

    file_path = DATA_DIR / country / f"{split}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["country"] = country

    return data


def build_training_example(
    query: str,
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a single training example in chat format.

    Returns a dict with 'messages' containing system, user, assistant
    turns. The system and user turns come from the same builder the
    Generator uses, so training and inference prompts match.
    """

    messages = build_messages(query, retrieved_docs)
    messages.append({"role": "assistant", "content": answer})

    return {"messages": messages}


def generate_training_data(
    countries: List[str],
    retriever: Retriever,
    sample_size: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Generate fine-tuning data by pairing questions with
    retrieved context and ground-truth answers.
    """

    training_examples = []

    for country in countries:
        # Load train + dev splits (test stays held out for evaluation)
        for split in ["train", "dev"]:
            try:
                records = load_split(country, split)
                logger.info(
                    f"Loaded {len(records)} {split} records "
                    f"for {country}"
                )
            except FileNotFoundError as e:
                logger.warning(f"Skipping: {e}")
                continue

            if sample_size and sample_size > 0:
                records = records[:sample_size]
                logger.info(
                    f"Subsampled to {len(records)} records "
                    f"for {country}/{split}"
                )

            for idx, record in enumerate(records):
                query = (
                    record.get("question")
                    or record.get("input_query", "")
                )
                answer = record.get("answer", "")

                if not query or not answer:
                    continue

                try:
                    # Retrieve one extra so that dropping the record's
                    # own KB entry still leaves top_k documents.
                    candidates = retriever.retrieve(
                        query, top_k=top_k + 1
                    )
                    retrieved_docs = [
                        doc for doc in candidates
                        if not (
                            doc["metadata"].get("source") == country
                            and doc["metadata"].get("data_id")
                            == record.get("data_id")
                        )
                    ][:top_k]
                except Exception as e:
                    logger.warning(
                        f"Retrieval failed for record {idx}: {e}"
                    )
                    retrieved_docs = []

                example = build_training_example(
                    query, answer, retrieved_docs
                )
                training_examples.append(example)

                if (idx + 1) % 500 == 0:
                    logger.info(
                        f"  [{country}/{split}] "
                        f"Processed {idx + 1}/{len(records)}"
                    )

    return training_examples


def save_training_data(
    examples: List[Dict[str, Any]],
    output_file: Path,
) -> Path:
    """Save training examples as chat-format JSONL."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(
        f"Saved {len(examples)} training examples to {output_file}"
    )

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build chat-format fine-tuning data from a knowledge "
            "base and the train + dev splits."
        )
    )
    parser.add_argument(
        "--kb",
        choices=["qa", "country_pdf"],
        default="qa",
        help=(
            "Knowledge base that supplies retrieved context "
            "(default: qa)"
        ),
    )
    parser.add_argument(
        "--country",
        choices=["all"] + SUPPORTED_COUNTRIES,
        default="all",
        help="Country dataset to use (default: all)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help=(
            "Samples per country per split "
            "(default: 0 = all data)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Documents to retrieve per query (default: {DEFAULT_TOP_K})",
    )

    args = parser.parse_args()

    countries = (
        SUPPORTED_COUNTRIES
        if args.country == "all"
        else [args.country]
    )
    sample_size = None if args.sample_size == 0 else args.sample_size

    # 1. Connect to the chosen knowledge base
    if args.kb == "qa":
        kb_path, kb_collection = QA_KB_PATH, QA_KB_COLLECTION
        build_cmd = "python -m scripts.build_qa_kb"
    else:
        kb_path = COUNTRY_KB_PATH
        kb_collection = COUNTRY_KB_COLLECTION
        build_cmd = "python -m scripts.build_country_kb"

    logger.info("Loading embedding model...")
    embedding_model = EmbeddingModel()

    logger.info(f"Connecting to '{args.kb}' KB at: {kb_path}")
    vector_store = VectorStore(
        db_path=kb_path,
        collection_name=kb_collection,
    )

    doc_count = vector_store.count()
    logger.info(f"'{args.kb}' KB has {doc_count} documents.")

    if doc_count == 0:
        logger.error(
            f"'{args.kb}' KB is empty! Run '{build_cmd}' first."
        )
        return

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    # 2. Generate and save training data
    logger.info("Generating training data...")
    examples = generate_training_data(
        countries=countries,
        retriever=retriever,
        sample_size=sample_size,
        top_k=args.top_k,
    )

    if not examples:
        logger.error("No training examples generated.")
        return

    save_training_data(examples, TRAINING_DATA_FILE)

    print("\n" + "=" * 60)
    print("          FINE-TUNING DATA SUMMARY")
    print("=" * 60)
    print(f"Countries used      : {', '.join(countries)}")
    print(f"Retrieval KB        : {args.kb}")
    print(f"Training examples   : {len(examples)}")
    print(f"Training data file  : {TRAINING_DATA_FILE}")
    print("Next                : python -m scripts.train_lora")
    print("=" * 60)


if __name__ == "__main__":
    main()
