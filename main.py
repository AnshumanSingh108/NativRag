from src.data_loader import load_all_data
from src.document_processor import process_documents


data = load_all_data("data")


for source, splits in data.items():

    train_records = splits["train"]

    output_path = f"processed_data/{source}_train.json"

    documents = process_documents(
        train_records,
        output_path
    )

    assert len(train_records) == len(documents), (
        f"{source}: record/document count mismatch! "
        f"{len(train_records)} records vs "
        f"{len(documents)} documents"
    )

    print(
        f"{source}: "
        f"{len(train_records)} records → "
        f"{len(documents)} documents"
    )