"""
Interactive CLI for the Nativ RAG system.

Connects to the Country PDF knowledge base by default and uses
the fine-tuned Ollama model for answer generation.

Usage:
    python main.py                              # Country PDF KB + fine-tuned model
    python main.py --kb qa                  # QA-pair KB + base model
    python main.py --llm-model llama3.2         # Override the LLM model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    COUNTRY_KB_PATH,
    COUNTRY_KB_COLLECTION,
    QA_KB_PATH,
    QA_KB_COLLECTION,
    BASE_LLM_MODEL,
    FINETUNED_LLM_MODEL,
    DEFAULT_TOP_K,
)
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator, resolve_model


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Nativ RAG Q&A system."
    )
    parser.add_argument(
        "--kb",
        choices=["country_pdf", "qa"],
        default="country_pdf",
        help="Knowledge base to use (default: country_pdf)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Ollama model name (auto-selected based on KB)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Documents to retrieve (default: {DEFAULT_TOP_K})",
    )

    args = parser.parse_args()

    # Select KB config
    if args.kb == "country_pdf":
        db_path = COUNTRY_KB_PATH
        collection = COUNTRY_KB_COLLECTION
        # Use the fine-tuned model if it exists, else the base model.
        # A model named with --llm-model is used as-is.
        llm_model = args.llm_model or resolve_model(FINETUNED_LLM_MODEL)
    else:
        db_path = QA_KB_PATH
        collection = QA_KB_COLLECTION
        llm_model = args.llm_model or BASE_LLM_MODEL

    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    print(f"Connecting to ChromaDB ({args.kb})...")
    vector_store = VectorStore(
        db_path=db_path,
        collection_name=collection,
    )

    # Check if the vector store is populated
    try:
        count = vector_store.count()
        if count == 0:
            if args.kb == "country_pdf":
                print(
                    "\nWARNING: Country KB is empty. "
                    "Run 'python -m scripts.build_country_kb' first."
                )
            else:
                print(
                    "\nWARNING: QA-pair KB is empty. "
                    "Run 'python -m scripts.build_qa_kb' first."
                )
            return
        print(f"Connected to ChromaDB with {count} documents.")
    except Exception as e:
        print(f"\nError checking vector store: {e}")
        return

    print("Initializing retriever and generator...")
    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )
    generator = Generator(model_name=llm_model)

    print("\n" + "=" * 50)
    print("          NATIV RAG SYSTEM READY")
    print(f"   KB: {args.kb} | LLM: {llm_model}")
    print("=" * 50)

    while True:
        query = input(
            "\nEnter your question (or 'quit' to exit): "
        ).strip()

        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            continue

        print("\nSearching knowledge base...")
        try:
            results = retriever.retrieve(
                query, top_k=args.top_k
            )
            print(f"Found {len(results)} relevant documents.")

            print(f"\nGenerating answer with {llm_model}...")
            answer = generator.generate(query, results)

            print("\n" + "=" * 50)
            print("ANSWER:")
            print(answer)
            print("=" * 50)

            print("\nSources used:")
            seen = set()
            for doc in results:
                meta = doc["metadata"]
                source = meta.get(
                    "source_file",
                    meta.get("source_answer_url", "Unknown"),
                )
                if source not in seen:
                    print(f"- {source}")
                    seen.add(source)

        except Exception as e:
            print(f"\nError processing query: {e}")


if __name__ == "__main__":
    main()
