from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever


embedding_model = EmbeddingModel()

vector_store = VectorStore(
    db_path="test_chroma_db",
    collection_name="test_collection",
)

retriever = Retriever(
    vector_store=vector_store,
    embedding_model=embedding_model,
)


query = "How can I apply for a tourist visa to Bangladesh?"

results = retriever.retrieve(
    query,
    top_k=2,
)


print("\nQuery:")
print(query)

print("\nRetrieved documents:")

for i, result in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")

    print("Distance:")
    print(result["distance"])

    print("\nMetadata:")
    print(result["metadata"])

    print("\nText:")
    print(result["text"])