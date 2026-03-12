import json
from pathlib import Path


def extract_text_from_transcript(transcript_path: Path) -> str:
    """Load transcript JSON, return text field or joined segments."""
    with open(transcript_path) as f:
        data = json.load(f)
    return data.get('text', '') or ' '.join(s['text'] for s in data.get('segments', []))
