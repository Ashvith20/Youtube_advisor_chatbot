import re
import os
from typing import List, Dict


def time_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm to seconds (float)."""
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt_transcript(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    segments = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp line
        if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", line):
            start_str, end_str = line.split(" --> ")
            start = time_to_seconds(start_str)
            end = time_to_seconds(end_str)

            # Read next lines until a blank line
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            segment_text = " ".join(text_lines)
            segments.append({
                "start": start,
                "end": end,
                "text": segment_text,
                "source": os.path.basename(file_path)
            })

        i += 1  # move to next line

    return segments


def parse_all_transcripts(transcript_dir: str = "transcripts") -> List[Dict]:
    """Parse all transcript .txt files in the transcripts folder."""
    all_segments = []
    for filename in os.listdir(transcript_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(transcript_dir, filename)
            print(f"Parsing {filename}")
            segments = parse_vtt_transcript(file_path)
            all_segments.extend(segments)
            print(f"   ↳ {len(segments)} segments parsed.")
    return all_segments


if __name__ == "__main__":
    all_segments = parse_all_transcripts("transcripts")
    print(f"\n Total segments parsed: {len(all_segments)}")
    print("First segment:")
    print(all_segments[0])
