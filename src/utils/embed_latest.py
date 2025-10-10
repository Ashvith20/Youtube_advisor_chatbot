import os
import pickle
from dotenv import load_dotenv
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from chunk import chunk_segments
from parser import parse_all_transcripts
from preprocessor import preprocess_all_segments

load_dotenv()

def embed_chunks(chunks: List[Dict], model_name="all-MiniLM-L6-v2") -> List[Dict]:
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return [
        {
            "embedding": emb.tolist(),
            "text": chunk["text"],
            "start": chunk["start"],
            "end": chunk["end"],
            "source": chunk["source"],
        }
        for emb, chunk in zip(embeddings, chunks)
    ]

def save_embeddings_to_file(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f" Saved {len(data)} embeddings to {path}")

def load_embeddings_from_file(path: str) -> List[Dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f" Loaded {len(data)} embeddings from {path}")
    return data

if __name__ == "__main__":
    cache_file = "data/embeddings_all.pkl"

    # Optional: delete cache if FORCE_REBUILD=1 is set in .env
    if os.getenv("FORCE_REBUILD") == "1" and os.path.exists(cache_file):
        print(" Removing old cached embeddings...")
        os.remove(cache_file)

    if os.path.exists(cache_file):
        data = load_embeddings_from_file(cache_file)
    else:
        #  Load, clean, chunk, embed both transcripts
        print(" Loading transcripts...")
        segments = parse_all_transcripts("transcripts")

        print(" Preprocessing segments...")
        cleaned = preprocess_all_segments(segments)

        print(" Chunking...")
        chunks = chunk_segments(cleaned, max_words=200)

        print(" Embedding...")
        data = embed_chunks(chunks)

        save_embeddings_to_file(data, cache_file)

    #  Sample output
    print(" Sample embedded chunk:")
    print(f"Text: {data[0]['text'][:60]}...")
    print(f"Source: {data[0]['source']}")
    print(f"Start time: {data[0]['start']} seconds")
    print(f"First 5 dims of embedding: {data[0]['embedding'][:5]}")
