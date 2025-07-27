#!/usr/bin/env python3
"""
Simulate the exact chunks that would be sent to LLM for Episode 8
"""
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from predictions.prediction_tracker.llm_extractor_optimized import OptimizedChunkProcessor

def main():
    # Load Episode 8 transcript
    transcript_path = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts/Bitcoin Dive Bar EP 8 - Tim B  Be Scarce - Sideways Forever BTC SMLR MSTR_yo6hikbIp5c_transcript.json")
    
    if not transcript_path.exists():
        print(f"Error: Episode 8 transcript not found at {transcript_path}")
        return
    
    print("="*80)
    print("SIMULATING EPISODE 8 CHUNK PROCESSING")
    print("="*80)
    
    # Load transcript
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Get the full text
    full_text = transcript_data.get('text', '')
    segments = transcript_data.get('segments', [])
    
    print(f"\nTranscript length: {len(full_text)} characters")
    print(f"Total segments: {len(segments)}")
    
    # Create chunk processor with current settings
    chunk_processor = OptimizedChunkProcessor(chunk_size=400000, overlap=20000)
    
    # Process chunks
    chunks = list(chunk_processor.create_chunks_generator(full_text))
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Find which chunk contains the "meta-planet below nine dollars" text
    target_phrase = "meta-planet below nine dollars"
    
    for i, chunk_info in enumerate(chunks):
        chunk_text = chunk_info['text']
        start_char = chunk_info['start_char']
        end_char = chunk_info['end_char']
        
        print(f"\n{'='*80}")
        print(f"CHUNK {i}")
        print(f"{'='*80}")
        print(f"Character range: {start_char:,} - {end_char:,}")
        print(f"Chunk size: {len(chunk_text):,} characters")
        print(f"Estimated tokens: ~{len(chunk_text) // 4} tokens")
        
        # Check if contains our target phrase
        if target_phrase in chunk_text.lower():
            print(f"\n>>> FOUND TARGET PHRASE IN THIS CHUNK! <<<")
            
            # Find position
            pos = chunk_text.lower().find(target_phrase)
            print(f"Position in chunk: {pos}")
            
            # Show context around the phrase
            context_start = max(0, pos - 500)
            context_end = min(len(chunk_text), pos + 500)
            
            print(f"\nContext around '{target_phrase}':")
            print("-" * 80)
            context = chunk_text[context_start:context_end]
            # Highlight the phrase
            context = context.replace("meta-planet below nine dollars", "**META-PLANET BELOW NINE DOLLARS**")
            print(context)
            print("-" * 80)
        
        # Check for other key terms
        chunk_lower = chunk_text.lower()
        key_terms = {
            'wntr': 0,
            'winter': 0,
            'inverse': 0,
            'meta planet': 0,
            'metaplanet': 0,
            'meta-planet': 0,
            'smlr': 0,
            'similar': 0,
            'nine dollar': 0,
            '9 dollar': 0,
            'below nine': 0,
            'below 9': 0
        }
        
        for term in key_terms:
            count = chunk_lower.count(term)
            key_terms[term] = count
        
        # Show non-zero counts
        found_terms = {k: v for k, v in key_terms.items() if v > 0}
        if found_terms:
            print(f"\nKey term occurrences in chunk:")
            for term, count in sorted(found_terms.items(), key=lambda x: x[1], reverse=True):
                print(f"  - '{term}': {count} times")
        
        # Save the chunk that contains our target
        if target_phrase in chunk_text.lower():
            output_file = f"episode8_chunk{i}_with_target.txt"
            with open(output_file, 'w') as f:
                f.write(f"CHUNK {i} - CONTAINS TARGET PHRASE\n")
                f.write(f"Character range: {start_char:,} - {end_char:,}\n")
                f.write(f"Size: {len(chunk_text):,} characters\n")
                f.write("="*80 + "\n\n")
                f.write("FULL CHUNK TEXT:\n")
                f.write("="*80 + "\n")
                f.write(chunk_text)
                f.write("\n\n" + "="*80 + "\n")
                f.write("KEY TERM OCCURRENCES:\n")
                f.write("="*80 + "\n")
                for term, count in sorted(found_terms.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{term}: {count} times\n")
            
            print(f"\nSaved chunk to: {output_file}")
            
            # Also check how much context before the target is included
            print(f"\nContext analysis:")
            print(f"- Characters before target phrase: {pos:,}")
            print(f"- Characters after target phrase: {len(chunk_text) - pos - len(target_phrase):,}")
            
            # Look for earlier mentions of relevant context
            earlier_context_terms = ['wntr', 'winter', 'inverse', 'msty']
            print(f"\nSearching for earlier context terms before the target...")
            for term in earlier_context_terms:
                term_pos = chunk_text[:pos].lower().rfind(term)
                if term_pos != -1:
                    distance = pos - term_pos
                    print(f"  - Found '{term}' {distance:,} characters before target")
                    # Show that context
                    term_context_start = max(0, term_pos - 100)
                    term_context_end = min(pos, term_pos + 100)
                    print(f"    Context: ...{chunk_text[term_context_start:term_context_end]}...")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total transcript: {len(full_text):,} characters")
    print(f"Chunk size: 400,000 characters (with 20,000 overlap)")
    print(f"Total chunks needed: {len(chunks)}")
    
    if len(chunks) == 1:
        print("\nThe entire transcript fits in a single chunk!")
        print("This means the LLM sees the FULL context of the episode.")
    else:
        print("\nThe transcript requires multiple chunks.")
        print("With 20,000 character overlap, there should be good context preservation.")

if __name__ == "__main__":
    main()