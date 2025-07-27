#!/usr/bin/env python3
"""
Debug script to run Stage 1 on a single episode
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Check if we're in venv
import os
if 'VIRTUAL_ENV' not in os.environ:
    print("Please activate virtual environment first: source venv/bin/activate")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predictions.prediction_tracker.llm_extractor_two_stage import TwoStageLLMExtractor


def main():
    # Get episode number from command line or default to EP 5
    episode_num = sys.argv[1] if len(sys.argv) > 1 else "5"
    
    # Create debug output directory
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    
    # Find the transcript file
    transcript_dir = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts")
    transcript_files = list(transcript_dir.glob(f"*EP {episode_num}*_transcript.json"))
    
    if not transcript_files:
        transcript_files = list(transcript_dir.glob(f"*EP {episode_num.zfill(2)}*_transcript.json"))
    
    if not transcript_files:
        print(f"No transcript found for episode {episode_num}")
        print("\nAvailable episodes:")
        for f in sorted(transcript_dir.glob("*_transcript.json")):
            print(f"  {f.name}")
        return
    
    transcript_path = transcript_files[0]
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = debug_dir / f"stage1_debug_ep{episode_num}_{timestamp}.json"
    
    print(f"Processing: {transcript_path.name}")
    print(f"Output will be saved to: {output_file}")
    print("="*80)
    
    # Load transcript
    with open(transcript_path, 'r') as f:
        data = json.load(f)
        text = data.get('text', '')
    
    episode_name = transcript_path.stem.replace('_transcript', '')
    print(f"Episode: {episode_name}")
    print(f"Transcript length: {len(text):,} characters")
    
    # Create extractor
    extractor = TwoStageLLMExtractor()
    
    # Run Stage 1 - find prediction locations
    print("\n[STAGE 1] Finding prediction locations...")
    print("(This will show the actual API calls and responses)")
    
    locations = extractor.find_prediction_locations(text, {'title': episode_name})
    
    if not locations:
        print("\nNo prediction locations found!")
        return
    
    print(f"\nFound {len(locations)} prediction locations")
    
    # Extract and display snippets
    results = {
        'episode': episode_name,
        'transcript_length': len(text),
        'locations_found': len(locations),
        'snippets': []
    }
    
    for i, loc in enumerate(locations):
        position = loc.get('position', 0)
        print(f"\n{'='*60}")
        print(f"LOCATION {i+1}: Character position {position}")
        
        # Extract snippet
        snippet = extractor.extract_snippet_around_position(text, position)
        
        # Show the snippet
        print(f"Snippet range: {snippet['char_start']}-{snippet['char_end']} ({len(snippet['text'])} chars)")
        print(f"Timestamp: {snippet.get('timestamp', 'unknown')}")
        print(f"\nFULL SNIPPET TEXT:")
        print("-" * 40)
        print(snippet['text'])
        print("-" * 40)
        
        # Highlight the prediction area (200 chars around position)
        rel_pos = position - snippet['char_start']
        start = max(0, rel_pos - 100)
        end = min(len(snippet['text']), rel_pos + 100)
        print(f"\nPREDICTION AREA (±100 chars from position):")
        print(f"...{snippet['text'][start:end]}...")
        
        # Save to results
        results['snippets'].append({
            'location_index': i + 1,
            'position': position,
            'snippet_range': f"{snippet['char_start']}-{snippet['char_end']}",
            'snippet_length': len(snippet['text']),
            'snippet_text': snippet['text'],
            'timestamp': snippet.get('timestamp', 'unknown'),
            'prediction_area': snippet['text'][start:end]
        })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    
    # Token usage
    print(f"\n=== TOKEN USAGE ===")
    print(f"Stage 1 (gpt-4.1-nano):")
    print(f"  Input tokens: {extractor.token_usage['snippet_extraction']['input']:,}")
    print(f"  Output tokens: {extractor.token_usage['snippet_extraction']['output']:,}")
    print(f"  API calls: {extractor.token_usage['snippet_extraction']['calls']}")
    
    # Cost estimate
    stage1_input_cost = extractor.token_usage['snippet_extraction']['input'] * 0.00003 / 1000
    stage1_output_cost = extractor.token_usage['snippet_extraction']['output'] * 0.00012 / 1000
    print(f"\nEstimated cost: ${stage1_input_cost + stage1_output_cost:.4f}")


if __name__ == "__main__":
    main()