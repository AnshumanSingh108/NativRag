"""
Evaluate Nativ RAG performance on test datasets.

Computes BERTScore (Precision, Recall, F1), ROUGE-1 F1 and BLEU
comparing RAG-generated answers against ground-truth answers from
test.json. These are the metrics used in the MultiNativQA paper.

Supports both the QA-pair KB and the Country PDF KB,
and both the base and fine-tuned Ollama models.

Usage:
    # Fine-tuned model (falls back to the base model if not installed)
    # with the QA-pair KB, on the entire test set of all 3 countries
    python -m scripts.evaluate --kb qa --sample-size 0

    # Baseline: base model, same KB and test set
    python -m scripts.evaluate --kb qa --sample-size 0 --llm-model llama3.2

    # Country PDF KB instead
    python -m scripts.evaluate --kb country_pdf --sample-size 0

    # Quick check on one country
    python -m scripts.evaluate --country india --sample-size 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Add project root to path for config import
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
    BERT_SCORE_MODEL,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_TOP_K,
    EVALUATION_RESULTS_DIR,
)
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator, resolve_model
from src.metrics import rouge1_f1, sentence_bleu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_split(country: str) -> List[Dict[str, Any]]:
    """Load test records for a given country."""

    file_path = DATA_DIR / country / "test.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["country"] = country

    return data


def _normalize(text: str) -> str:
    """Casefold and collapse whitespace for substring matching."""
    return " ".join(text.casefold().split())


def answer_in_context(
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
) -> bool:
    """
    True if the reference answer appears verbatim in a retrieved doc.

    With the QA-pair KB this usually means a training record with the
    same answer was retrieved, so the model can copy the reference
    instead of answering from knowledge. Reporting the two groups
    separately shows how much of the score comes from that.
    """

    needle = _normalize(answer)
    if not needle:
        return False

    return any(
        needle in _normalize(doc["text"]) for doc in retrieved_docs
    )


def get_kb_config(kb_name: str) -> tuple:
    """Return (db_path, collection_name) for the named KB."""

    if kb_name == "country_pdf":
        return COUNTRY_KB_PATH, COUNTRY_KB_COLLECTION
    elif kb_name == "qa":
        return QA_KB_PATH, QA_KB_COLLECTION
    else:
        raise ValueError(
            f"Unknown KB '{kb_name}'. "
            f"Use 'country_pdf' or 'qa'."
        )


def run_evaluation(
    countries: List[str],
    kb_name: str = "country_pdf",
    sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE,
    top_k: int = DEFAULT_TOP_K,
    bert_model: str = BERT_SCORE_MODEL,
    llm_model: str = FINETUNED_LLM_MODEL,
    output_file: Optional[str] = "evaluation_results.json",
):
    """
    Run RAG generation on test samples and calculate BERT-F1 scores.
    """

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.error(
            "bert-score is not installed. "
            "Install with: pip install bert-score"
        )
        return

    # 1. Initialize RAG components
    db_path, collection_name = get_kb_config(kb_name)

    logger.info(f"Using KB: {kb_name} ({db_path})")
    logger.info(f"Using LLM: {llm_model}")

    logger.info("Initializing Embedding Model...")
    embedding_model = EmbeddingModel()

    logger.info(f"Connecting to ChromaDB ({collection_name})...")
    vector_store = VectorStore(
        db_path=db_path,
        collection_name=collection_name,
    )

    doc_count = vector_store.count()
    logger.info(f"Connected to ChromaDB with {doc_count} documents.")

    if doc_count == 0:
        if kb_name == "country_pdf":
            logger.error(
                "Country KB is empty! "
                "Run 'python -m scripts.build_country_kb' first."
            )
        else:
            logger.error(
                "QA-pair KB is empty! "
                "Run 'python -m scripts.build_qa_kb' first."
            )
        return

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )
    generator = Generator(model_name=llm_model)

    # 2. Collect test samples
    all_samples: List[Dict[str, Any]] = []

    for country in countries:
        records = load_test_split(country)
        logger.info(
            f"Loaded {len(records)} test samples for {country}"
        )
        if sample_size and sample_size > 0:
            records = records[:sample_size]
            logger.info(
                f"Subsampled to {len(records)} "
                f"test samples for {country}"
            )
        all_samples.extend(records)

    logger.info(
        f"Total test samples to evaluate: {len(all_samples)}"
    )

    predictions: List[str] = []
    references: List[str] = []
    detailed_records: List[Dict[str, Any]] = []

    # 3. Generate answers
    for idx, sample in enumerate(all_samples, 1):
        query = (
            sample.get("question")
            or sample.get("input_query", "")
        )
        ground_truth = sample.get("answer", "")

        logger.info(
            f"[{idx}/{len(all_samples)}] "
            f"Query ({sample['country']}): {query[:70]}..."
        )

        try:
            retrieved_docs = retriever.retrieve(
                query, top_k=top_k
            )
            generated_answer = generator.generate(
                query, retrieved_docs
            )
        except Exception as e:
            logger.error(
                f"Error generating answer for query "
                f"'{query}': {e}"
            )
            generated_answer = ""
            retrieved_docs = []

        predictions.append(generated_answer)
        references.append(ground_truth)

        # Build source info (support both PDF and QA sources)
        retrieved_sources = []
        for doc in retrieved_docs:
            meta = doc["metadata"]
            source = meta.get(
                "source_file",
                meta.get("source_answer_url", ""),
            )
            retrieved_sources.append(source)

        detailed_records.append({
            "data_id": sample.get("data_id"),
            "country": sample.get("country"),
            "query": query,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "retrieved_sources": retrieved_sources,
            "answer_in_context": answer_in_context(
                ground_truth, retrieved_docs
            ),
        })

    # 4. Calculate BERTScore
    logger.info(
        f"Computing BERTScore using model: {bert_model}..."
    )
    P, R, F1 = bert_score_fn(
        cands=predictions,
        refs=references,
        model_type=bert_model,
        verbose=True,
    )

    p_scores = P.numpy().tolist()
    r_scores = R.numpy().tolist()
    f1_scores = F1.numpy().tolist()

    for i, item in enumerate(detailed_records):
        item["bert_precision"] = float(p_scores[i])
        item["bert_recall"] = float(r_scores[i])
        item["bert_f1"] = float(f1_scores[i])

    mean_f1 = float(np.mean(f1_scores))
    mean_p = float(np.mean(p_scores))
    mean_r = float(np.mean(r_scores))

    # 4b. Lexical metrics (ROUGE-1 F1 and sentence BLEU, 0-1 scale)
    logger.info("Computing ROUGE-1 and BLEU...")
    for item in detailed_records:
        item["rouge1_f1"] = rouge1_f1(
            item["generated_answer"], item["ground_truth"]
        )
        item["bleu"] = sentence_bleu(
            item["generated_answer"], item["ground_truth"]
        )

    mean_rouge1 = float(
        np.mean([item["rouge1_f1"] for item in detailed_records])
    )
    mean_bleu = float(
        np.mean([item["bleu"] for item in detailed_records])
    )

    # 5. Breakdown by country
    country_f1: Dict[str, List[float]] = {}
    country_rouge1: Dict[str, List[float]] = {}
    country_bleu: Dict[str, List[float]] = {}
    for item in detailed_records:
        c = item["country"]
        country_f1.setdefault(c, []).append(item["bert_f1"])
        country_rouge1.setdefault(c, []).append(item["rouge1_f1"])
        country_bleu.setdefault(c, []).append(item["bleu"])

    # 6. Breakdown by whether the reference answer was retrieved
    overlap_f1: Dict[str, List[float]] = {
        "answer_in_context": [],
        "answer_not_in_context": [],
    }
    for item in detailed_records:
        key = (
            "answer_in_context"
            if item["answer_in_context"]
            else "answer_not_in_context"
        )
        overlap_f1[key].append(item["bert_f1"])

    print("\n" + "=" * 60)
    print("                 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Knowledge Base          : {kb_name}")
    print(f"LLM Model               : {llm_model}")
    print(f"Total Evaluated Samples : {len(all_samples)}")
    print(f"BERTScore Model         : {bert_model}")
    print(f"Mean BERT Precision     : {mean_p:.4f}")
    print(f"Mean BERT Recall        : {mean_r:.4f}")
    print(f"Mean BERT F1 Score      : {mean_f1:.4f}")
    print(f"Mean ROUGE-1 F1         : {mean_rouge1:.4f}")
    print(f"Mean BLEU               : {mean_bleu:.4f}")
    print("-" * 60)
    print(
        f"  {'Per-country':<14}{'BERT-F1':>9}{'BLEU':>9}"
        f"{'ROUGE-1':>9}{'n':>8}"
    )
    for country, scores in country_f1.items():
        print(
            f"  - {country.capitalize():<12}"
            f"{np.mean(scores):>9.4f}"
            f"{np.mean(country_bleu[country]):>9.4f}"
            f"{np.mean(country_rouge1[country]):>9.4f}"
            f"{len(scores):>8}"
        )
    print("-" * 60)
    print("BERT-F1 by whether the reference answer was retrieved:")
    for label, scores in overlap_f1.items():
        mean = f"{np.mean(scores):.4f}" if scores else "   n/a"
        print(f"  - {label:<22}: {mean} (n={len(scores)})")
    print("=" * 60)

    # 7. Save results to disk
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "config": {
                "knowledge_base": kb_name,
                "llm_model": llm_model,
                "bert_model": bert_model,
                "top_k": top_k,
            },
            "summary": {
                "total_samples": len(all_samples),
                "mean_precision": mean_p,
                "mean_recall": mean_r,
                "mean_f1": mean_f1,
                "mean_rouge1_f1": mean_rouge1,
                "mean_bleu": mean_bleu,
                "country_f1": {
                    k: float(np.mean(v))
                    for k, v in country_f1.items()
                },
                "country_rouge1_f1": {
                    k: float(np.mean(v))
                    for k, v in country_rouge1.items()
                },
                "country_bleu": {
                    k: float(np.mean(v))
                    for k, v in country_bleu.items()
                },
                "overlap_f1": {
                    k: {
                        "mean_f1": (
                            float(np.mean(v)) if v else None
                        ),
                        "n": len(v),
                    }
                    for k, v in overlap_f1.items()
                },
            },
            "samples": detailed_records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                result_payload, f, ensure_ascii=False, indent=2
            )
        logger.info(
            f"Detailed evaluation saved to {out_path.resolve()}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Nativ RAG performance on test datasets "
            "and compute BERT-F1, ROUGE-1 and BLEU."
        )
    )
    parser.add_argument(
        "--country",
        choices=["all"] + SUPPORTED_COUNTRIES,
        default="all",
        help="Country dataset to evaluate on (default: all)",
    )
    parser.add_argument(
        "--kb",
        choices=["country_pdf", "qa"],
        default="country_pdf",
        help=(
            "Knowledge base to use: "
            "'country_pdf' (PDF-based, default) or "
            "'qa' (QA train-based)"
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            f"Samples to evaluate per country "
            f"(default: {DEFAULT_SAMPLE_SIZE}, use 0 for all)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Documents to retrieve per query (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default=BERT_SCORE_MODEL,
        help=f"BERTScore model (default: {BERT_SCORE_MODEL})",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=(
            "Ollama model name. Defaults to the fine-tuned model "
            f"('{FINETUNED_LLM_MODEL}') if installed, otherwise "
            f"the base model ('{BASE_LLM_MODEL}')."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to save evaluation output JSON (default: "
            "evaluation_results/<kb>_<model>.json, so runs with "
            "different models do not overwrite each other)"
        ),
    )

    args = parser.parse_args()

    countries = (
        SUPPORTED_COUNTRIES
        if args.country == "all"
        else [args.country]
    )
    sample_limit = (
        None if args.sample_size == 0 else args.sample_size
    )

    # Fine-tuned model if installed, otherwise the base model.
    # Pass --llm-model llama3.2 explicitly for a baseline run.
    if args.llm_model:
        llm_model = args.llm_model
    else:
        llm_model = resolve_model(FINETUNED_LLM_MODEL)

    output_file = args.output or str(
        EVALUATION_RESULTS_DIR
        / f"{args.kb}_{llm_model.replace(':', '-')}.json"
    )

    run_evaluation(
        countries=countries,
        kb_name=args.kb,
        sample_size=sample_limit,
        top_k=args.top_k,
        bert_model=args.bert_model,
        llm_model=llm_model,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
