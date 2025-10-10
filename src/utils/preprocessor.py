import re
from typing import Dict, List


def preprocess_segment(segment: Dict) -> Dict:
    """
    Clean and normalize a single transcript segment.
    Returns a new segment dict with cleaned text.
    """
    text = segment["text"]

    # Remove non-speech tags like [Music], [Laughter]
    text = re.sub(r"\[.*?\]", "", text)

    # Remove filler words
    text = re.sub(r"\b(uh+|um+|erm+|ah+|like)\b", "", text, flags=re.IGNORECASE)

    # Remove extra spaces and normalize punctuation
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("’", "'")  # normalize apostrophes
    text = text.strip(" .!?")

    # Optionally filter out very short segments (handled outside this function)

    return {
        **segment,
        "text": text
    }


def preprocess_all_segments(segments: List[Dict], min_words: int = 3) -> List[Dict]:
    """
    Apply preprocessing to all segments.
    Filters out very short or empty segments after cleaning.
    """
    cleaned = []
    for seg in segments:
        processed = preprocess_segment(seg)
        if len(processed["text"].split()) >= min_words:
            cleaned.append(processed)
    return cleaned


# Example usage
if __name__ == "__main__":
    from parser import parse_all_transcripts

    print(" Loading parsed segments...")
    segments = parse_all_transcripts("transcripts")

    print(f" Total before cleaning: {len(segments)}")
    cleaned_segments = preprocess_all_segments(segments)
    print(f" Total after cleaning: {len(cleaned_segments)}")

    print("\n Sample cleaned segment:")
    print(cleaned_segments[0])

    print("\n Checking metadata preservation for first 5 segments:")
    for i in range(min(5, len(segments))):
        original = segments[i]
        cleaned = cleaned_segments[i]
        print(f"\nSegment {i + 1}:")
        print(f" Original start: {original['start']} | Cleaned start: {cleaned['start']}")
        print(f" Original end: {original['end']} | Cleaned end: {cleaned['end']}")
        print(f" Original source: {original['source']} | Cleaned source: {cleaned['source']}")
        print(f" Original text: {original['text']}")
        print(f" Cleaned text: {cleaned['text']}")

