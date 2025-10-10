from typing import List, Dict
from parser import parse_all_transcripts
from preprocessor import preprocess_all_segments  # import your preprocessing function


def chunk_segments(
    segments: List[Dict],
    max_words: int = 200
) -> List[Dict]:
    """
    Groups transcript segments into larger chunks based on a word limit.
    Preserves original metadata for citation.
    """
    chunks = []
    current_chunk = []
    current_word_count = 0
    chunk_start = None

    for segment in segments:
        words = segment["text"].split()
        word_count = len(words)

        # Start tracking start time
        if not current_chunk:
            chunk_start = segment["start"]

        # If adding this would exceed the limit, finish the current chunk
        if current_word_count + word_count > max_words:
            combined_text = " ".join(seg["text"] for seg in current_chunk)
            chunks.append({
                "text": combined_text,
                "source": current_chunk[0]["source"],
                "start": chunk_start,
                "end": current_chunk[-1]["end"]
            })
            current_chunk = []
            current_word_count = 0
            chunk_start = segment["start"]

        current_chunk.append(segment)
        current_word_count += word_count

    # Add any remaining chunk
    if current_chunk:
        combined_text = " ".join(seg["text"] for seg in current_chunk)
        chunks.append({
            "text": combined_text,
            "source": current_chunk[0]["source"],
            "start": chunk_start,
            "end": current_chunk[-1]["end"]
        })

    return chunks


if __name__ == "__main__":
    import pprint

    print(" Loading and parsing all transcripts...")
    all_segments = parse_all_transcripts("transcripts")
    print(f" Total segments loaded: {len(all_segments)}")

    print("🧹 Preprocessing all segments...")
    cleaned_segments = preprocess_all_segments(all_segments)
    print(f" Segments after cleaning: {len(cleaned_segments)}")

    chunks = chunk_segments(cleaned_segments, max_words=200)
    print(f"Chunked into {len(chunks)} total chunks")

    print("\n First chunk sample:")
    pprint.pprint(chunks[0])
