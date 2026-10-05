import json
from pathlib import Path

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


PROCESSED_DATA_DIR = Path("processed_data")
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "nativ_rag"

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
    """Build the ChromaDB knowledge base from processed training data."""

    all_documents = []

    for file_name in DATA_FILES:

        file_path = PROCESSED_DATA_DIR / file_name

        documents = load_processed_documents(file_path)

        print(
            f"Loaded {len(documents)} documents "
            f"from {file_name}"
        )

        all_documents.extend(documents)

    if not all_documents:
        raise ValueError("No documents found to index.")

    print(
        f"\nTotal documents to index: "
        f"{len(all_documents)}"
    )

    print("\nLoading embedding model...")
    embedding_model = EmbeddingModel()

    print("\nGenerating embeddings...")
    embeddings = embedding_model.generate(all_documents)

    print("\nConnecting to ChromaDB...")
    vector_store = VectorStore(
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    print("\nAdding documents to ChromaDB...")
    vector_store.add_documents(
        all_documents,
        embeddings,
    )

    stored_count = vector_store.count()

    print("\nIndexing complete!")
    print(f"Documents expected: {len(all_documents)}")
    print(f"Documents in ChromaDB: {stored_count}")

    assert stored_count == len(all_documents), (
        f"ChromaDB count mismatch! "
        f"Expected {len(all_documents)}, "
        f"found {stored_count}"
    )

    print("\nKnowledge base sanity check passed!")


if __name__ == "__main__":
    main()