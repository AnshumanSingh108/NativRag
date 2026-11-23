import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_TOP_K


class Retriever:
    """Retrieve relevant documents from the vector store."""

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query, top_k=DEFAULT_TOP_K):
        """
        Retrieve the most relevant documents for a user query.

        Args:
            query: User's question as a string.
            top_k: Maximum number of documents to retrieve.

        Returns:
            List of retrieved documents with metadata and distance.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_document = [{"text": query}]

        query_embedding = self.embedding_model.generate(
            query_document
        )[0]

        results = self.vector_store.search(
            query_embedding,
            n_results=top_k,
        )

        retrieved_documents = []

        for i, document in enumerate(results["documents"][0]):
            retrieved_documents.append({
                "text": document,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return retrieved_documents
