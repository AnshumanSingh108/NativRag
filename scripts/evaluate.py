import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
SUPPORTED_COUNTRIES = ["bangladesh", "india", "qatar"]


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


def run_evaluation(
    countries: List[str],
    sample_size: Optional[int] = 50,
    top_k: int = 5,
    bert_model: str = "bert-base-multilingual-cased",
    llm_model: str = "llama3.2",
    output_file: Optional[str] = "evaluation_results.json",
):
    """
    Run RAG generation on test samples and calculate BERT-F1 scores.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.error(
            "bert-score is not installed. Please install it with: pip install bert-score"
        )
        return

    # 1. Initialize RAG components
    logger.info("Initializing Embedding Model...")
    embedding_model = EmbeddingModel()

    logger.info("Connecting to ChromaDB...")
    vector_store = VectorStore(db_path="chroma_db", collection_name="nativ_rag")
    doc_count = vector_store.count()
    logger.info(f"Connected to ChromaDB with {doc_count} documents.")

    if doc_count == 0:
        logger.error("ChromaDB is empty! Please run 'python -m scripts.build_index' first.")
        return

    retriever = Retriever(vector_store=vector_store, embedding_model=embedding_model)
    generator = Generator(model_name=llm_model)

    # 2. Collect test samples
    all_samples: List[Dict[str, Any]] = []
    for country in countries:
        records = load_test_split(country)
        logger.info(f"Loaded {len(records)} test samples for {country}")
        if sample_size and sample_size > 0:
            records = records[:sample_size]
            logger.info(f"Subsampled {len(records)} test samples for {country}")
        all_samples.extend(records)

    logger.info(f"Total test samples to evaluate: {len(all_samples)}")

    predictions: List[str] = []
    references: List[str] = []
    detailed_records: List[Dict[str, Any]] = []

    # 3. Generate answers
    for idx, sample in enumerate(all_samples, 1):
        query = sample.get("question") or sample.get("input_query", "")
        ground_truth = sample.get("answer", "")

        logger.info(f"[{idx}/{len(all_samples)}] Query ({sample['country']}): {query[:70]}...")

        try:
            retrieved_docs = retriever.retrieve(query, top_k=top_k)
            generated_answer = generator.generate(query, retrieved_docs)
        except Exception as e:
            logger.error(f"Error generating answer for query '{query}': {e}")
            generated_answer = ""
            retrieved_docs = []

        predictions.append(generated_answer)
        references.append(ground_truth)

        detailed_records.append({
            "data_id": sample.get("data_id"),
            "country": sample.get("country"),
            "query": query,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "retrieved_sources": [
                doc["metadata"].get("source_answer_url", "")
                for doc in retrieved_docs
            ],
        })

    # 4. Calculate BERTScore
    logger.info(f"Computing BERTScore using model: {bert_model}...")
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

    # 5. Breakdown by country
    country_f1: Dict[str, List[float]] = {}
    for item in detailed_records:
        c = item["country"]
        country_f1.setdefault(c, []).append(item["bert_f1"])

    print("\n" + "=" * 60)
    print("                 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Samples : {len(all_samples)}")
    print(f"BERTScore Model         : {bert_model}")
    print(f"Mean BERT Precision     : {mean_p:.4f}")
    print(f"Mean BERT Recall        : {mean_r:.4f}")
    print(f"Mean BERT F1 Score      : {mean_f1:.4f}")
    print("-" * 60)
    print("Per-Country BERT-F1 Breakdown:")
    for country, scores in country_f1.items():
        print(f"  - {country.capitalize():<12}: {np.mean(scores):.4f} (n={len(scores)})")
    print("=" * 60)

    # 6. Save results to disk
    if output_file:
        out_path = Path(output_file)
        result_payload = {
            "summary": {
                "total_samples": len(all_samples),
                "bert_model": bert_model,
                "mean_precision": mean_p,
                "mean_recall": mean_r,
                "mean_f1": mean_f1,
                "country_f1": {k: float(np.mean(v)) for k, v in country_f1.items()},
            },
            "samples": detailed_records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed evaluation saved to {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Nativ RAG performance on test datasets and compute BERT-F1 score."
    )
    parser.add_argument(
        "--country",
        choices=["all"] + SUPPORTED_COUNTRIES,
        default="all",
        help="Country dataset to evaluate on (default: all)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of samples to evaluate per country (default: 50, use 0 for all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-multilingual-cased",
        help="Pretrained model for BERTScore (default: bert-base-multilingual-cased)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3.2",
        help="Ollama model name (default: llama3.2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation output JSON (default: evaluation_results.json)",
    )

    args = parser.parse_args()

    countries = SUPPORTED_COUNTRIES if args.country == "all" else [args.country]
    sample_limit = None if args.sample_size == 0 else args.sample_size

    run_evaluation(
        countries=countries,
        sample_size=sample_limit,
        top_k=args.top_k,
        bert_model=args.bert_model,
        llm_model=args.llm_model,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
