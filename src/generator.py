import logging
import sys
from pathlib import Path

import ollama

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BASE_LLM_MODEL, GENERATION_TEMPERATURE

logger = logging.getLogger(__name__)


# Single source of truth for the prompt. Used both for answering
# (Generator) and for building fine-tuning data (scripts/finetune_llm.py),
# so a tuned model sees the same format it was trained on.
SYSTEM_PROMPT = (
    "You are a knowledgeable and helpful assistant that answers "
    "questions about countries, their laws, constitutions, culture, "
    "and civic procedures using the provided reference documents.\n\n"
    "Instructions:\n"
    "- Use the information from the documents to give a "
    "clear, detailed, and helpful answer.\n"
    "- Combine information from multiple documents if needed.\n"
    "- The documents may be in English, Hindi, or Bengali. "
    "Answer in the same language as the user's question.\n"
    "- If the documents contain step-by-step instructions, "
    "present them clearly.\n"
    "- Only say you cannot answer if the documents are truly "
    "unrelated to the question."
)


def format_context(retrieved_documents):
    """Format retrieved documents as numbered, source-labelled blocks."""

    context_texts = []

    for i, doc in enumerate(retrieved_documents, 1):
        # Support both PDF-sourced and QA-sourced documents
        source = doc["metadata"].get(
            "source_file",
            doc["metadata"].get("source_answer_url", "Unknown"),
        )
        context_texts.append(
            f"--- Document {i} (Source: {source}) ---\n"
            f"{doc['text']}"
        )

    return "\n\n".join(context_texts)


def build_messages(query, retrieved_documents):
    """Build the system + user chat messages for one question."""

    user_message = (
        f"Reference Documents:\n"
        f"{format_context(retrieved_documents)}\n\n"
        f"Question: {query}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _installed_models():
    """Return the set of model names Ollama has installed."""

    names = set()

    for model in ollama.list().models:
        names.add(model.model)
        # "llama3.2:latest" is also reachable as "llama3.2"
        if model.model.endswith(":latest"):
            names.add(model.model[: -len(":latest")])

    return names


def resolve_model(preferred, fallback=BASE_LLM_MODEL):
    """
    Return `preferred` if Ollama has it installed, else `fallback`.

    Used for default model choices only: a model the user names
    explicitly should fail loudly instead of being swapped.
    """

    try:
        installed = _installed_models()
    except Exception as e:
        logger.warning(f"Could not list Ollama models ({e}).")
        return preferred

    if preferred in installed:
        return preferred

    logger.warning(
        f"Ollama model '{preferred}' is not installed; "
        f"falling back to '{fallback}'. "
        "Register a trained adapter with "
        "'python -m scripts.finetune_llm --adapter <path>'."
    )
    return fallback


class Generator:
    """Generate final answers using Ollama and retrieved context."""

    def __init__(
        self,
        model_name=BASE_LLM_MODEL,
        temperature=GENERATION_TEMPERATURE,
    ):
        self.model_name = model_name
        self.temperature = temperature

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

        response = ollama.chat(
            model=self.model_name,
            messages=build_messages(query, retrieved_documents),
            options={"temperature": self.temperature},
        )

        return response["message"]["content"]
