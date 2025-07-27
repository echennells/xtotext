#!/usr/bin/env python3
"""
Debug script to see full snippet extraction
"""
import json
import sys
from pathlib import Path

# Check if we're in venv
import os
if 'VIRTUAL_ENV' not in os.environ:
    print("Please activate virtual environment first: source venv/bin/activate")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predictions.prediction_tracker.llm_extractor_two_stage import TwoStageLLMExtractor

def main():
    # Get Episode 5 transcript
    transcript_path = "data/episodes/bitcoin_dive_bar_analysis/transcripts/Bitcoin Dive Bar EP 5 - Bitcoin 120k  Guests BTC SMLR MSTR_vuA8LrC_z3s_transcript.json"
    
    with open(transcript_path, 'r') as f:
        data = json.load(f)
        text = data.get('text', '')
    
    # Create extractor
    extractor = TwoStageLLMExtractor()
    
    # Find prediction locations
    print("Finding prediction locations...")
    locations = extractor.find_prediction_locations(text, {'title': 'EP 5'})
    
    print(f"\nFound {len(locations)} locations:")
    for i, loc in enumerate(locations):
        print(f"  Location {i+1}: {loc}")
    
    # Extract snippets
    print("\n\nExtracting snippets around each location:")
    for i, loc in enumerate(locations[:3]):  # Just first 3 to save space
        position = loc.get('position', 0)
        snippet = extractor.extract_snippet_around_position(text, position)
        
        print(f"\n{'='*80}")
        print(f"LOCATION {i+1}: Position {position}")
        print(f"Snippet range: {snippet['char_start']}-{snippet['char_end']} ({len(snippet['text'])} chars)")
        print(f"\nFULL SNIPPET TEXT:")
        print("-" * 40)
        print(snippet['text'])
        print("-" * 40)
        
        # Show the prediction in context (100 chars before/after position)
        rel_pos = position - snippet['char_start']
        start = max(0, rel_pos - 100)
        end = min(len(snippet['text']), rel_pos + 100)
        print(f"\nPREDICTION CONTEXT (position {position}):")
        print(f"...{snippet['text'][start:end]}...")

if __name__ == "__main__":
    main()