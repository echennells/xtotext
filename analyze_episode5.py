#!/usr/bin/env python3
"""Analyze Episode 5 prediction extraction"""
import json
from pathlib import Path

# Episode 5 transcript
transcript_file = "data/episodes/bitcoin_dive_bar_analysis/transcripts/Bitcoin Dive Bar EP 5 - Bitcoin 120k  Guests BTC SMLR MSTR_vuA8LrC_z3s_transcript.json"

if Path(transcript_file).exists():
    print("Analyzing Episode 5 prediction extraction...")
    
    # Load transcript
    with open(transcript_file, 'r') as f:
        data = json.load(f)
        text = data.get('text', '')
    
    # Find the specific prediction context
    search_phrase = "tomorrow similar is going to go to 50 dollars"
    
    if search_phrase in text.lower():
        pos = text.lower().find(search_phrase)
        
        # Show context around the prediction
        context_start = max(0, pos - 500)
        context_end = min(len(text), pos + 500)
        
        print(f"\nFound prediction at position {pos}")
        print(f"\nContext ({context_end - context_start} chars):")
        print("=" * 80)
        print(text[context_start:context_end])
        print("=" * 80)
        
        # Check timestamp
        if 'segments' in data:
            # Find which segment contains this position
            char_count = 0
            for seg in data['segments']:
                seg_text = seg.get('text', '')
                if char_count <= pos < char_count + len(seg_text):
                    print(f"\nSegment timestamp: {seg.get('start', 0)}s = {int(seg.get('start', 0))//60}:{int(seg.get('start', 0))%60:02d}")
                    break
                char_count += len(seg_text)
    else:
        print(f"Prediction phrase not found in transcript")
        
        # Search for variations
        for phrase in ["similar is going to", "50 dollars", "50 bucks", "tomorrow"]:
            count = text.lower().count(phrase)
            print(f"Found '{phrase}': {count} times")
            
        # Find where "50 dollars" appears
        print("\nSearching for '50 dollars' contexts...")
        search_text = text.lower()
        pos = 0
        while True:
            pos = search_text.find("50 dollar", pos)
            if pos == -1:
                break
            # Show context
            context_start = max(0, pos - 100)
            context_end = min(len(text), pos + 100)
            print(f"\nAt position {pos}:")
            print(text[context_start:context_end])
            pos += 1
else:
    print(f"Transcript not found: {transcript_file}")