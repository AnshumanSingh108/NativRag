import sys
from pathlib import Path

import chromadb

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COUNTRY_KB_PATH, COUNTRY_KB_COLLECTION


class VectorStore:
    """Manage the ChromaDB vector store."""

    def __init__(
        self,
        db_path=COUNTRY_KB_PATH,
        collection_name=COUNTRY_KB_COLLECTION,
    ):
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, documents, embeddings):
        """Add documents, embeddings, and metadata to ChromaDB."""

        if not documents:
            raise ValueError("No documents provided.")

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match "
                "number of embeddings."
            )

        # Create globally unique ChromaDB IDs
        ids = [
            f"{document['metadata']['source']}_"
            f"{document['metadata']['data_id']}"
            for document in documents
        ]

        # Ensure IDs are unique
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate ChromaDB IDs detected. "
                "The combination of source and data_id "
                "must be unique."
            )

        texts = [document["text"] for document in documents]

        # Sanitize metadata: ChromaDB only accepts
        # str, int, float, bool values
        metadatas = []
        for document in documents:
            clean_meta = {}
            for k, v in document["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        # ChromaDB has a max batch size limit (5461).
        # We batch our inserts to stay under the limit.
        batch_size = 5000
        embeddings_list = embeddings.tolist()

        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                embeddings=embeddings_list[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def count(self):
        """Return the number of documents in the collection."""
        return self.collection.count()

    def search(self, query_embedding, n_results=5):
        """
        Search the vector store for documents similar
        to a query embedding.
        """

        collection_count = self.collection.count()

        if collection_count == 0:
            raise ValueError("Vector store is empty.")

        if n_results < 1:
            raise ValueError("n_results must be at least 1.")

        n_results = min(n_results, collection_count)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
        )

        return results
