"""
Prepare fine-tuning data and package a fine-tuned model for Ollama.

Ollama cannot train models. Its Modelfile ADAPTER line loads a LoRA
adapter that was already trained elsewhere (safetensors or GGUF).
So fine-tuning is two steps:

1. This script builds chat-format training data:
   loads train + dev data from data/{country}/, retrieves context
   for each question from the chosen KB (--kb qa, the default, or
   country_pdf), and writes finetune_data/training_data.jsonl. The
   prompt format is the same one src/generator.py uses at answer
   time.
2. Train a LoRA adapter on that JSONL with scripts/train_lora.py
   (needs a CUDA GPU), then run this script again with --adapter to
   register it in Ollama as the fine-tuned model.

Without --adapter the script only writes the training data. It does
not create a model, so 'nativrag-finetuned' never points at an
untuned copy of the base model.

With the QA-pair KB, a training question's own record is in the KB
(it was indexed from the train split). Retrieving it would put the
ground-truth answer in the prompt, so the record itself is excluded
from its own retrieved context. Dev records are not in the KB.

Usage:
    python -m scripts.finetune_llm                       # training data only
    python -m scripts.finetune_llm --country india --sample-size 500
    python -m scripts.finetune_llm --adapter path/to/adapter --skip-data
"""

import argparse
import json
import logging
import subprocess
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
    BASE_LLM_MODEL,
    FINETUNED_LLM_MODEL,
    FINETUNE_OUTPUT_DIR,
    GENERATION_TEMPERATURE,
    MODELFILE_PATH,
    DEFAULT_TOP_K,
)
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import SYSTEM_PROMPT, build_messages

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
    retrieved PDF context and ground-truth answers.
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


def create_modelfile(
    base_model: str,
    adapter_path: Path,
    modelfile_path: Path,
) -> Path:
    """
    Create an Ollama Modelfile that applies a trained LoRA adapter.

    `adapter_path` must be a LoRA adapter trained on `base_model`
    (a safetensors directory or a GGUF file), not training data.
    """

    modelfile_content = f"""FROM {base_model}
ADAPTER {adapter_path.resolve()}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature {GENERATION_TEMPERATURE}
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
"""

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    logger.info(f"Created Modelfile at {modelfile_path}")

    return modelfile_path


def run_ollama_create(
    model_name: str,
    modelfile_path: Path,
) -> bool:
    """Run 'ollama create' to register the fine-tuned model."""

    cmd = [
        "ollama", "create",
        model_name,
        "-f", str(modelfile_path),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        logger.error(
            "Ollama CLI not found. "
            "Please install Ollama: https://ollama.com/download"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Ollama create timed out after 600 seconds.")
        return False

    if result.returncode != 0:
        logger.error(
            f"Failed to create model. "
            f"Return code: {result.returncode}"
        )
        if result.stderr:
            logger.error(f"stderr: {result.stderr}")
        return False

    logger.info(f"Successfully created model '{model_name}'")
    if result.stdout:
        logger.info(f"stdout: {result.stdout}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build fine-tuning data from a knowledge base and "
            "train+dev data, and register a trained LoRA adapter "
            "in Ollama."
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
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_LLM_MODEL,
        help=(
            f"Base Ollama model the adapter was trained on "
            f"(default: {BASE_LLM_MODEL})"
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=FINETUNED_LLM_MODEL,
        help=f"Fine-tuned model name (default: {FINETUNED_LLM_MODEL})",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help=(
            "Path to a LoRA adapter trained on --base-model "
            "(safetensors directory or .gguf file). When given, "
            "the adapter is registered in Ollama as --model-name."
        ),
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help=(
            "Don't regenerate training data "
            "(use with --adapter after training)"
        ),
    )

    args = parser.parse_args()

    adapter_path = Path(args.adapter) if args.adapter else None
    if adapter_path and not adapter_path.exists():
        parser.error(f"Adapter not found: {adapter_path}")

    if args.skip_data and not adapter_path:
        parser.error("--skip-data only makes sense with --adapter")

    countries = (
        SUPPORTED_COUNTRIES
        if args.country == "all"
        else [args.country]
    )
    sample_size = None if args.sample_size == 0 else args.sample_size

    num_examples = None

    if not args.skip_data:
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

        num_examples = len(examples)
        logger.info(f"Generated {num_examples} training examples.")

        save_training_data(examples, TRAINING_DATA_FILE)

    # 3. Register a trained adapter in Ollama, if one was given
    created = False

    if adapter_path:
        modelfile_path = create_modelfile(
            base_model=args.base_model,
            adapter_path=adapter_path,
            modelfile_path=MODELFILE_PATH,
        )
        created = run_ollama_create(args.model_name, modelfile_path)
    else:
        logger.info(
            "No --adapter given, so no Ollama model was created. "
            "Train a LoRA adapter with 'python -m scripts.train_lora' "
            "on a GPU, then rerun with --adapter <path> --skip-data."
        )

    print("\n" + "=" * 60)
    print("            FINE-TUNING SUMMARY")
    print("=" * 60)
    print(f"Countries used      : {', '.join(countries)}")
    print(f"Retrieval KB        : {args.kb}")
    if num_examples is not None:
        print(f"Training examples   : {num_examples}")
    print(f"Training data file  : {TRAINING_DATA_FILE}")
    print(f"Base model          : {args.base_model}")
    if adapter_path:
        print(f"Adapter             : {adapter_path}")
        print(f"Modelfile           : {MODELFILE_PATH}")
    if created:
        print(f"Fine-tuned model    : {args.model_name}")
    else:
        print("Fine-tuned model    : not created")
    print("=" * 60)


if __name__ == "__main__":
    main()
