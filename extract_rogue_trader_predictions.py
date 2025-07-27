#!/usr/bin/env python3
"""
Extract predictions from Rogue Trader episodes using the prompts we generated
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from llm.llm_client import OpenAIClient as LLMClient
from predictions.prediction_tracker.llm_extractor_custom_prompts import CustomPromptExtractor


def main():
    # Load the generated prompts
    # prompts_file = Path("data/episodes/rogue_trader_analysis/latest_prompts.json")
    prompts_file = Path("data/episodes/rogue_trader_analysis/improved_prompts.json")
    
    if not prompts_file.exists():
        print("Error: No generated prompts found. Run generate_prompts_from_analysis.py first!")
        return
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    print("="*80)
    print("EXTRACTING PREDICTIONS FROM ROGUE TRADER EPISODES")
    print("="*80)
    print("\nUsing prompts generated from Rogue Trader analysis")
    
    # Get all Rogue Trader transcripts
    transcripts_dir = Path("data/episodes/rogue_trader_analysis/transcripts")
    transcript_files = list(transcripts_dir.glob("*_transcript.json"))
    
    if not transcript_files:
        print(f"\nError: No transcripts found in {transcripts_dir}")
        return
    
    # Sort for consistent ordering
    transcript_files.sort(key=lambda x: x.name)
    
    print(f"\nFound {len(transcript_files)} Rogue Trader episodes to analyze:")
    for tf in transcript_files:
        print(f"  - {tf.stem}")
    
    # Initialize the extractor with custom prompts
    print("\n\nInitializing prediction extractor with custom prompts...")
    extractor = CustomPromptExtractor(
        stage1_prompt=prompts['stage1'],
        stage2_prompt=prompts['stage2']
    )
    
    # Process each episode
    all_results = []
    total_predictions = 0
    
    for i, transcript_file in enumerate(transcript_files):
        print(f"\n{'='*80}")
        print(f"Episode {i+1}/{len(transcript_files)}: {transcript_file.stem}")
        print('='*80)
        
        # Load transcript
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
            text = transcript_data.get('text', '')
        
        print(f"Transcript length: {len(text)} characters")
        
        # Extract predictions
        print("Extracting predictions...")
        try:
            predictions = extractor.extract_predictions_from_file(
                transcript_file=str(transcript_file),
                episode_info={
                    'title': transcript_file.stem.replace('_', ' '),
                    'filename': transcript_file.name,
                    'video_id': transcript_file.stem.split('_')[-2] if '_' in transcript_file.stem else ''
                }
            )
            
            print(f"✓ Found {len(predictions)} predictions")
            
            # Show first few predictions as preview
            if predictions:
                print("\nPreview of predictions:")
                for j, pred in enumerate(predictions[:3]):
                    print(f"\n  Prediction {j+1}:")
                    print(f"    Asset: {pred.get('asset', 'N/A')}")
                    print(f"    Price: {pred.get('price', 'N/A')}")
                    print(f"    Date/Timeframe: {pred.get('date', 'N/A')}")
                    print(f"    Context: {pred.get('context', '')[:150]}...")
                
                if len(predictions) > 3:
                    print(f"\n  ... and {len(predictions) - 3} more predictions")
            
            # Store results
            episode_result = {
                'episode': transcript_file.stem,
                'file': transcript_file.name,
                'predictions_count': len(predictions),
                'predictions': predictions,
                'processed_at': datetime.now().isoformat()
            }
            
            all_results.append(episode_result)
            total_predictions += len(predictions)
            
        except Exception as e:
            print(f"✗ Error extracting predictions: {e}")
            episode_result = {
                'episode': transcript_file.stem,
                'file': transcript_file.name,
                'predictions_count': 0,
                'predictions': [],
                'error': str(e),
                'processed_at': datetime.now().isoformat()
            }
            all_results.append(episode_result)
    
    # Save all results
    output_dir = Path("data/episodes/rogue_trader_analysis/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    detailed_file = output_dir / f"rogue_trader_predictions_{timestamp}.json"
    with open(detailed_file, 'w') as f:
        json.dump({
            'extraction_date': datetime.now().isoformat(),
            'episodes_processed': len(all_results),
            'total_predictions': total_predictions,
            'prompts_used': {
                'stage1': prompts['stage1'][:200] + '...',
                'stage2': prompts['stage2'][:200] + '...'
            },
            'results': all_results
        }, f, indent=2)
    
    # Save summary
    summary_file = output_dir / f"rogue_trader_predictions_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("ROGUE TRADER PREDICTION EXTRACTION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Episodes Processed: {len(all_results)}\n")
        f.write(f"Total Predictions Found: {total_predictions}\n")
        f.write(f"Average Predictions per Episode: {total_predictions/len(all_results):.1f}\n\n")
        
        f.write("PREDICTIONS BY EPISODE:\n")
        f.write("-"*50 + "\n")
        for result in all_results:
            f.write(f"{result['episode']}: {result['predictions_count']} predictions\n")
    
    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nProcessed {len(all_results)} episodes")
    print(f"Total predictions found: {total_predictions}")
    print(f"Average per episode: {total_predictions/len(all_results):.1f}")
    
    print(f"\nResults saved to:")
    print(f"  - Detailed: {detailed_file}")
    print(f"  - Summary: {summary_file}")
    
    # Show which episodes had the most predictions
    sorted_results = sorted(all_results, key=lambda x: x['predictions_count'], reverse=True)
    print(f"\nTop episodes by prediction count:")
    for result in sorted_results[:5]:
        print(f"  - {result['episode']}: {result['predictions_count']} predictions")


if __name__ == "__main__":
    main()