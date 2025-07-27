#!/usr/bin/env python3
"""
Debug script to run only Stage 1 prediction location finding
and save all found snippets to a file
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
    # Create debug output directory
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    
    # Output file for snippets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = debug_dir / f"stage1_snippets_{timestamp}.json"
    
    # Get all transcript files
    transcript_dir = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts")
    transcript_files = list(transcript_dir.glob("*_transcript.json"))
    
    print(f"Found {len(transcript_files)} transcript files")
    print(f"Output will be saved to: {output_file}")
    print("="*80)
    
    all_results = []
    
    # Create extractor
    extractor = TwoStageLLMExtractor()
    
    for transcript_path in transcript_files:
        print(f"\nProcessing: {transcript_path.name}")
        
        # Load transcript
        with open(transcript_path, 'r') as f:
            data = json.load(f)
            text = data.get('text', '')
            
        # Extract episode info from filename
        episode_name = transcript_path.stem.replace('_transcript', '')
        
        print(f"Transcript length: {len(text):,} characters")
        
        # Run Stage 1 only - find prediction locations
        print("\n[STAGE 1] Finding prediction locations...")
        try:
            locations = extractor.find_prediction_locations(text, {'title': episode_name})
            
            if not locations:
                print("No prediction locations found")
                continue
                
            print(f"Found {len(locations)} prediction locations")
            
            # Extract snippets for each location
            episode_snippets = {
                'episode': episode_name,
                'transcript_length': len(text),
                'locations_found': len(locations),
                'snippets': []
            }
            
            for i, loc in enumerate(locations):
                position = loc.get('position', 0)
                print(f"\n  Location {i+1}: Position {position}")
                
                # Extract snippet around this position
                snippet = extractor.extract_snippet_around_position(text, position)
                
                # Add to results
                snippet_data = {
                    'location_index': i + 1,
                    'position': position,
                    'snippet_range': f"{snippet['char_start']}-{snippet['char_end']}",
                    'snippet_length': len(snippet['text']),
                    'snippet_text': snippet['text'],
                    'timestamp': snippet.get('timestamp', 'unknown')
                }
                
                episode_snippets['snippets'].append(snippet_data)
                
                # Print preview of snippet (first 200 chars)
                preview = snippet['text'][:200].replace('\n', ' ')
                print(f"  Preview: {preview}...")
                
            all_results.append(episode_snippets)
            
        except Exception as e:
            print(f"Error processing {episode_name}: {e}")
            continue
    
    # Save all results
    print(f"\n{'='*80}")
    print(f"Saving results to {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump({
            'debug_run': timestamp,
            'total_episodes': len(transcript_files),
            'episodes_with_predictions': len(all_results),
            'total_snippets': sum(ep['locations_found'] for ep in all_results),
            'results': all_results
        }, f, indent=2)
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Processed {len(transcript_files)} episodes")
    print(f"- Found predictions in {len(all_results)} episodes")
    print(f"- Total snippets extracted: {sum(ep['locations_found'] for ep in all_results)}")
    print(f"\nFull results saved to: {output_file}")
    
    # Also print token usage
    print(f"\n=== TOKEN USAGE ===")
    print(f"Stage 1 (gpt-4.1-nano):")
    print(f"  Input tokens: {extractor.token_usage['snippet_extraction']['input']:,}")
    print(f"  Output tokens: {extractor.token_usage['snippet_extraction']['output']:,}")
    print(f"  API calls: {extractor.token_usage['snippet_extraction']['calls']}")
    
    # Calculate costs (using rough estimates)
    stage1_input_cost = extractor.token_usage['snippet_extraction']['input'] * 0.00003 / 1000
    stage1_output_cost = extractor.token_usage['snippet_extraction']['output'] * 0.00012 / 1000
    print(f"\nEstimated Stage 1 cost: ${stage1_input_cost + stage1_output_cost:.4f}")


if __name__ == "__main__":
    main()