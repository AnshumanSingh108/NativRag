from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator

def main():
    print("Loading embedding model...")
    embedding_model = EmbeddingModel()
    
    print("Connecting to ChromaDB...")
    vector_store = VectorStore(
        db_path="chroma_db",
        collection_name="nativ_rag"
    )
    
    # Check if the vector store is populated
    try:
        count = vector_store.count()
        if count == 0:
            print("\nWARNING: ChromaDB is empty. Please run 'python -m scripts.build_index' first.")
            return
        print(f"Connected to ChromaDB with {count} documents.")
    except Exception as e:
        print(f"\nError checking vector store: {e}")
        return
    
    print("Initializing retriever and generator...")
    retriever = Retriever(vector_store=vector_store, embedding_model=embedding_model)
    generator = Generator(model_name="llama3.2")
    
    print("\n" + "="*50)
    print("          NATIV RAG SYSTEM READY")
    print("="*50)
    
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        print("\nSearching knowledge base...")
        try:
            results = retriever.retrieve(query, top_k=5)
            print(f"Found {len(results)} relevant documents.")
            
            print("\nGenerating answer with Ollama (llama3.2)...")
            answer = generator.generate(query, results)
            
            print("\n" + "="*50)
            print("ANSWER:")
            print(answer)
            print("="*50)
            
            print("\nSources used:")
            seen_urls = set()
            for doc in results:
                url = doc['metadata'].get('source_answer_url', 'Unknown URL')
                if url not in seen_urls:
                    print(f"- {url}")
                    seen_urls.add(url)
                    
        except Exception as e:
            print(f"\nError processing query: {e}")

if __name__ == "__main__":
    main()