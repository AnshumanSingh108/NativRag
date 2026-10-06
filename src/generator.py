import ollama


class Generator:
    """Generate final answers using Ollama and retrieved context."""

    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def generate(self, query, retrieved_documents):
        """
        Generate an answer to a query based on retrieved documents.

        Args:
            query: The user's question.
            retrieved_documents: List of document dicts from the Retriever.

        Returns:
            The generated text string.
        """

        if not retrieved_documents:
            return (
                "I could not find any relevant information "
                "to answer your question."
            )

        context_texts = []

        for i, doc in enumerate(retrieved_documents, 1):
            source_url = doc["metadata"].get(
                "source_answer_url", "Unknown"
            )
            context_texts.append(
                f"--- Document {i} (Source: {source_url}) ---\n"
                f"{doc['text']}"
            )

        context = "\n\n".join(context_texts)

        prompt = (
            "You are a knowledgeable and helpful assistant "
            "that answers questions using the provided "
            "reference documents.\n\n"
            "Instructions:\n"
            "- Use the information from the documents below "
            "to give a clear, detailed, and helpful answer.\n"
            "- Combine information from multiple documents "
            "if needed.\n"
            "- The documents may be in English or Hindi. "
            "Answer in the same language as the user's "
            "question.\n"
            "- If the documents contain step-by-step "
            "instructions, present them clearly.\n"
            "- Only say you cannot answer if the documents "
            "are truly unrelated to the question.\n\n"
            f"Reference Documents:\n{context}\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
        )

        return response["response"]
