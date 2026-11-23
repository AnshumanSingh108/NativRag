import json
from pathlib import Path


REQUIRED_FIELDS = {
    "data_id",
    "category",
    "input_query",
    "question",
    "is_reliable",
    "answer",
    "source_answer_url",
}


def process_documents(records, output_path):
    """
    Convert dataset records into RAG documents and save them to disk.

    Each input record becomes exactly one RAG document.
    """

    documents = []

    for record in records:

        missing_fields = REQUIRED_FIELDS - record.keys()

        if missing_fields:
            raise ValueError(
                f"Record {record.get('data_id', '<unknown>')} "
                f"is missing fields: {missing_fields}"
            )

        text = (
            f"Input query:\n"
            f"{record['input_query']}\n\n"
            f"Question:\n"
            f"{record['question']}\n\n"
            f"Answer:\n"
            f"{record['answer']}"
        )

        metadata = {
            "data_id": record["data_id"],
            "category": record["category"],
            "is_reliable": record["is_reliable"],
            "source_answer_url": record["source_answer_url"],
            "source": record["source"],
            "split": record["split"],
        }

        documents.append({
            "text": text,
            "metadata": metadata,
        })

    if not documents:
        raise ValueError("No documents were created.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4
        )

    return documents