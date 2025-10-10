import os
from typing import List, Dict
from chromadb import  PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CACHE_PATH = "data/embeddings_all.pkl"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "transcripts_collection"

def load_embeddings_from_file(path: str) -> List[Dict]:
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f" Loaded {len(data)} embeddings from {path}")
    return data

class ChromaDBClient:
    def __init__(
        self,
        persist_directory: str = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        #  NEW WAY TO INSTANTIATE CLIENT
        self.client = PersistentClient(path=self.persist_directory)

        # Define the embedding function (only needed for queries, not for precomputed embedding ingestion)
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(name=self.collection_name)
            print(f" Loaded existing collection '{self.collection_name}'")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f" Created new collection '{self.collection_name}'")
        return collection

    def add_chunks(self, chunks: List[Dict]):
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk.get("source", ""),
                "start": chunk.get("start", 0),
                "end": chunk.get("end", 0),
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f" Added {len(chunks)} chunks to collection '{self.collection_name}'")

    def query(self, query_text: str, top_k: int = 3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\nTop {top_k} results for query: '{query_text}'")
        for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), 1):
            print(f"\nResult {i}:")
            print(f"Text snippet: {doc[:200]}...")
            print(f"Metadata: {meta}")
            print(f"Distance: {dist:.4f}")
            print("-" * 50)

        return results

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Load cached precomputed embeddings and metadata
    chunks = load_embeddings_from_file(CACHE_PATH)

    # Initialize Chroma client
    chroma_db = ChromaDBClient()

    # Add chunks
    chroma_db.add_chunks(chunks)

    # Run query
    chroma_db.query("How do I improve my video intro?", top_k=3)
