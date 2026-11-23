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


def load_json_file(file_path):
    """Load records from a single JSON file."""

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {file_path}")

    return data


def validate_record(record, file_path):
    """Check that a record contains all required fields."""

    missing_fields = REQUIRED_FIELDS - record.keys()

    if missing_fields:
        raise ValueError(
            f"Missing fields {missing_fields} in {file_path}"
        )


def load_source(source_path):
    """Load train, dev, and test data from one source."""

    source_path = Path(source_path)
    source_name = source_path.name

    splits = {}

    for split in ["train", "dev", "test"]:

        file_path = source_path / f"{split}.json"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {split}.json in {source_path}"
            )

        records = load_json_file(file_path)

        for record in records:
            validate_record(record, file_path)

            # Add metadata without modifying the original dataset fields
            record["source"] = source_name
            record["split"] = split

        splits[split] = records

    return splits


def load_all_data(data_dir):
    """Load all dataset sources from the data directory."""

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    all_data = {}

    for source_path in sorted(data_dir.iterdir()):

        if source_path.is_dir():
            all_data[source_path.name] = load_source(source_path)

    return all_data