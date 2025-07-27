#!/usr/bin/env python3
"""
Debug Episode 8 to see what context is being sent to LLM
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from predictions.prediction_tracker.llm_extractor_optimized import OptimizedLLMPredictionExtractor, OptimizedChunkProcessor

def main():
    # Load Episode 8 transcript
    transcript_path = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts/Bitcoin Dive Bar EP 8 - Tim B  Be Scarce - Sideways Forever BTC SMLR MSTR_yo6hikbIp5c_transcript.json")
    
    if not transcript_path.exists():
        print(f"Error: Episode 8 transcript not found at {transcript_path}")
        return
    
    print("="*80)
    print("EPISODE 8 DEBUG - Finding context around 52:56 (3176s)")
    print("="*80)
    
    # Load transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Get the full text
    full_text = transcript_data.get('text', '')
    segments = transcript_data.get('segments', [])
    
    print(f"\nTranscript length: {len(full_text)} characters")
    print(f"Total segments: {len(segments)}")
    
    # Find segments around 52:56 (3176s)
    target_time = 3176
    
    # Get 2 minutes of context before and after
    context_segments = []
    for seg in segments:
        start = seg.get('start', 0)
        if start >= (target_time - 120) and start <= (target_time + 120):
            context_segments.append(seg)
    
    print(f"\nFound {len(context_segments)} segments around timestamp 52:56")
    
    # Show the actual text around that timestamp
    print("\n" + "="*80)
    print("SEGMENTS AROUND 52:56 (±2 minutes)")
    print("="*80)
    
    for seg in context_segments:
        start = seg.get('start', 0)
        text = seg.get('text', '')
        start_min = int(start // 60)
        start_sec = int(start % 60)
        
        # Highlight segments very close to target
        if abs(start - target_time) < 10:
            print(f"\n>>> [{start_min}:{start_sec:02d}] {text}")
        else:
            print(f"\n[{start_min}:{start_sec:02d}] {text}")
    
    # Now simulate what the chunking system would do
    print("\n" + "="*80)
    print("SIMULATING CHUNK EXTRACTION")
    print("="*80)
    
    # Create chunk processor with current settings
    chunk_processor = OptimizedChunkProcessor(chunk_size=400000, overlap=20000)
    
    # Find which chunk would contain timestamp 3176
    chunk_num = 0
    for chunk_info in chunk_processor.create_chunks_generator(full_text):
        chunk_text = chunk_info['text']
        start_char = chunk_info['start_char']
        end_char = chunk_info['end_char']
        
        # Check if our target timestamp text is in this chunk
        # Look for text around "below nine dollars" or "WNTR"
        if any(phrase in chunk_text.lower() for phrase in ['wntr', 'winter', 'below nine', 'meta planet']):
            print(f"\nChunk {chunk_num} contains relevant content:")
            print(f"  Chunk size: {len(chunk_text)} chars")
            print(f"  Character range: {start_char} - {end_char}")
            
            # Find specific mentions
            chunk_lower = chunk_text.lower()
            
            # Search for WNTR/Winter mentions
            wntr_positions = []
            for term in ['wntr', 'winter']:
                pos = 0
                while True:
                    pos = chunk_lower.find(term, pos)
                    if pos == -1:
                        break
                    wntr_positions.append((term, pos))
                    pos += 1
            
            if wntr_positions:
                print(f"\n  Found {len(wntr_positions)} mentions of WNTR/Winter:")
                for term, pos in wntr_positions[:5]:  # Show first 5
                    context_start = max(0, pos - 100)
                    context_end = min(len(chunk_text), pos + 100)
                    print(f"    ...{chunk_text[context_start:context_end]}...")
            
            # Search for price mentions
            import re
            price_pattern = r'\$?\d+(?:\.\d+)?(?:\s*(?:dollars?|bucks?))?'
            price_matches = list(re.finditer(price_pattern, chunk_text, re.IGNORECASE))
            
            print(f"\n  Found {len(price_matches)} price mentions in chunk")
            
            # Save the chunk for inspection
            output_file = f"episode8_chunk{chunk_num}_debug.txt"
            with open(output_file, 'w') as f:
                f.write(f"CHUNK {chunk_num} CONTENTS:\n")
                f.write(f"Character range: {start_char} - {end_char}\n")
                f.write(f"Size: {len(chunk_text)} characters\n")
                f.write("="*80 + "\n\n")
                f.write(chunk_text)
            
            print(f"\n  Full chunk saved to: {output_file}")
            
            # Also extract a focused window around any "nine dollars" mention
            nine_dollar_pos = chunk_lower.find("nine dollar")
            if nine_dollar_pos != -1:
                # Get 1000 chars before and after
                window_start = max(0, nine_dollar_pos - 1000)
                window_end = min(len(chunk_text), nine_dollar_pos + 1000)
                
                print("\n" + "="*80)
                print("CONTEXT WINDOW AROUND 'NINE DOLLARS':")
                print("="*80)
                print(chunk_text[window_start:window_end])
                
                # Save focused context
                with open("episode8_nine_dollars_context.txt", 'w') as f:
                    f.write("CONTEXT AROUND 'NINE DOLLARS' MENTION:\n")
                    f.write("="*80 + "\n\n")
                    f.write(chunk_text[window_start:window_end])
        
        chunk_num += 1
    
    print("\n" + "="*80)
    print("LOOKING FOR WNTR/WINTER MENTIONS IN FULL TRANSCRIPT")
    print("="*80)
    
    # Search entire transcript for WNTR/Winter
    full_lower = full_text.lower()
    for term in ['wntr', 'winter', 'inverse']:
        positions = []
        pos = 0
        while True:
            pos = full_lower.find(term, pos)
            if pos == -1:
                break
            positions.append(pos)
            pos += 1
        
        if positions:
            print(f"\nFound '{term}' {len(positions)} times in transcript:")
            for p in positions[:3]:  # Show first 3
                context_start = max(0, p - 200)
                context_end = min(len(full_text), p + 200)
                print(f"  Position {p}: ...{full_text[context_start:context_end]}...")

if __name__ == "__main__":
    main()