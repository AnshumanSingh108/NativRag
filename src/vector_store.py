import chromadb


CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "nativ_rag"


class VectorStore:
    """Manage the ChromaDB vector store."""

    def __init__(
        self,
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
    ):
        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, documents, embeddings):
        """
        Add documents, embeddings, and metadata to ChromaDB.
        """

        if not documents:
            raise ValueError("No documents provided.")

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match "
                "number of embeddings."
            )

        # Create globally unique ChromaDB IDs using
        # source + original dataset data_id.
        ids = [
            f"{document['metadata']['source']}_"
            f"{document['metadata']['data_id']}"
            for document in documents
        ]

        # Make sure our generated ChromaDB IDs are unique.
        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate ChromaDB IDs detected. "
                "The combination of source and data_id "
                "must be unique."
            )

        texts = [
            document["text"]
            for document in documents
        ]

        metadatas = [
            document["metadata"]
            for document in documents
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
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
            raise ValueError(
                "n_results must be at least 1."
            )

        n_results = min(
            n_results,
            collection_count
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=n_results,
        )

        return results