#!/usr/bin/env python3
"""
Clear ALL episodes from processed list to force re-extraction
"""
import json
from pathlib import Path

episodes_file = Path("prediction_data/episodes.json")

# Save empty list
with open(episodes_file, 'w') as f:
    json.dump([], f, indent=2)

print("Cleared ALL episodes from processed list")
print("All episodes will now re-extract predictions when run")
print("\nNOTE: This will NOT re-transcribe the audio (transcripts already exist)")
print("It will only re-extract predictions from existing transcripts")