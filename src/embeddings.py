from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class EmbeddingModel:
    """Generate multilingual embeddings for RAG documents."""

    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def generate(self, documents):
        """
        Generate embeddings for a list of processed documents.

        Args:
            documents: List of dictionaries containing a "text" field.

        Returns:
            Embeddings as a NumPy array.
        """

        if not documents:
            raise ValueError("No documents provided for embedding.")

        texts = [document["text"] for document in documents]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings