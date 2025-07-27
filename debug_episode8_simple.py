#!/usr/bin/env python3
"""
Simple debug of Episode 8 to find WNTR/Winter mentions
"""
import json
import re
from pathlib import Path

def main():
    # Load Episode 8 transcript
    transcript_path = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts/Bitcoin Dive Bar EP 8 - Tim B  Be Scarce - Sideways Forever BTC SMLR MSTR_yo6hikbIp5c_transcript.json")
    
    if not transcript_path.exists():
        print(f"Error: Episode 8 transcript not found at {transcript_path}")
        return
    
    print("="*80)
    print("EPISODE 8 DEBUG - Finding WNTR/Winter mentions")
    print("="*80)
    
    # Load transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Get the full text
    full_text = transcript_data.get('text', '')
    segments = transcript_data.get('segments', [])
    
    print(f"\nTranscript length: {len(full_text)} characters")
    print(f"Total segments: {len(segments)}")
    
    # Search for WNTR/Winter/inverse mentions
    full_lower = full_text.lower()
    
    print("\n" + "="*80)
    print("SEARCHING FOR KEY TERMS")
    print("="*80)
    
    search_terms = ['wntr', 'winter', 'inverse', 'meta planet', 'metaplanet', 'nine dollar', '9 dollar']
    
    for term in search_terms:
        positions = []
        pos = 0
        while True:
            pos = full_lower.find(term, pos)
            if pos == -1:
                break
            positions.append(pos)
            pos += len(term)
        
        if positions:
            print(f"\nFound '{term}' {len(positions)} times:")
            for i, p in enumerate(positions):
                # Get surrounding context
                context_start = max(0, p - 300)
                context_end = min(len(full_text), p + 300)
                context = full_text[context_start:context_end]
                
                # Find which segment this is in
                char_count = 0
                segment_time = None
                for seg in segments:
                    seg_text = seg.get('text', '')
                    if char_count <= p < char_count + len(seg_text):
                        segment_time = seg.get('start', 0)
                        break
                    char_count += len(seg_text) + 1  # +1 for space between segments
                
                if segment_time:
                    mins = int(segment_time // 60)
                    secs = int(segment_time % 60)
                    print(f"\n  Occurrence {i+1} at [{mins}:{secs:02d}]:")
                else:
                    print(f"\n  Occurrence {i+1}:")
                
                # Highlight the term
                context = context.replace(term, f"**{term.upper()}**")
                print(f"    ...{context}...")
    
    # Now look specifically around the 52:56 timestamp
    print("\n" + "="*80)
    print("SEGMENTS AROUND 52:56 (3176s)")
    print("="*80)
    
    target_time = 3176
    
    # Get segments from 50:00 to 55:00
    for seg in segments:
        start = seg.get('start', 0)
        if 3000 <= start <= 3300:  # 50:00 to 55:00
            text = seg.get('text', '')
            mins = int(start // 60)
            secs = int(start % 60)
            
            # Check if contains our terms
            contains_terms = any(term in text.lower() for term in ['wntr', 'winter', 'nine', 'meta'])
            
            if contains_terms:
                print(f"\n>>> [{mins}:{secs:02d}] {text}")
            else:
                print(f"\n[{mins}:{secs:02d}] {text}")
    
    # Save full context around 52:56 for inspection
    print("\n" + "="*80)
    print("SAVING FULL CONTEXT")
    print("="*80)
    
    # Get all text from segments between 51:00 and 54:00
    context_segments = []
    for seg in segments:
        start = seg.get('start', 0)
        if 3060 <= start <= 3240:  # 51:00 to 54:00
            context_segments.append(seg)
    
    context_text = " ".join([seg.get('text', '') for seg in context_segments])
    
    with open("episode8_context_51_54.txt", 'w') as f:
        f.write("EPISODE 8 CONTEXT (51:00 - 54:00)\n")
        f.write("="*80 + "\n\n")
        f.write(context_text)
    
    print(f"Saved context to: episode8_context_51_54.txt")
    print(f"Context length: {len(context_text)} characters")

if __name__ == "__main__":
    main()