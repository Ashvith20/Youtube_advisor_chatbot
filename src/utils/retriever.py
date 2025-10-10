import chromadb
from chromadb.config import Settings

def get_chroma_client():
    return chromadb.Client(Settings(persist_directory="chroma_persist"))

def main():
    client = get_chroma_client()
    collection = client.get_collection("transcripts_collection")

    results = collection.get(include=["metadatas", "documents"], limit=3)

    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Text: {doc[:80]}...")
        print(f"Metadata: {meta}")

if __name__ == "__main__":
    main()
