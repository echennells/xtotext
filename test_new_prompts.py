#!/usr/bin/env python3
"""
Test the new prompts generated from Rogue Trader analysis on Bitcoin Dive Bar episodes
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
from predictions.prediction_tracker.llm_extractor_two_stage import TwoStageLLMExtractor


def main():
    # Load the generated prompts
    prompts_file = Path("data/episodes/rogue_trader_analysis/latest_prompts.json")
    
    if not prompts_file.exists():
        print("Error: No generated prompts found. Run generate_prompts_from_analysis.py first!")
        return
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    print("="*80)
    print("LOADED PROMPTS FROM ROGUE TRADER ANALYSIS")
    print("="*80)
    print("\nStage 1 Preview:")
    print(prompts['stage1'][:300] + "...")
    print("\nStage 2 Preview:")
    print(prompts['stage2'][:300] + "...")
    
    # Check for Bitcoin Dive Bar transcripts
    bitcoin_dive_bar_dir = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts")
    
    if not bitcoin_dive_bar_dir.exists():
        print(f"\nNo Bitcoin Dive Bar transcripts found at {bitcoin_dive_bar_dir}")
        print("You need to run main.py first to download and transcribe Bitcoin Dive Bar episodes")
        return
    
    transcript_files = list(bitcoin_dive_bar_dir.glob("*_transcript.json"))
    
    if not transcript_files:
        print(f"\nNo transcript files found in {bitcoin_dive_bar_dir}")
        print("You need to run main.py first to download and transcribe Bitcoin Dive Bar episodes")
        return
    
    print(f"\n\nFound {len(transcript_files)} Bitcoin Dive Bar transcripts to test")
    
    # Let user select which episode to test
    print("\nAvailable episodes:")
    for i, tf in enumerate(transcript_files):
        print(f"{i+1}. {tf.stem}")
    
    selection = input("\nEnter episode number to test (or 'all' for all episodes): ")
    
    if selection.lower() == 'all':
        episodes_to_test = transcript_files
    else:
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(transcript_files):
                episodes_to_test = [transcript_files[idx]]
            else:
                print("Invalid selection")
                return
        except:
            print("Invalid selection")
            return
    
    # Initialize the extractor with custom prompts
    print("\nInitializing prediction extractor with new prompts...")
    extractor = TwoStageLLMExtractor()
    
    # Override the default prompts with our generated ones
    extractor.stage1_prompt = prompts['stage1']
    extractor.stage2_prompt = prompts['stage2']
    
    print(f"\nTesting on {len(episodes_to_test)} episode(s)...\n")
    
    # Test each selected episode
    all_predictions = []
    
    for transcript_file in episodes_to_test:
        print("="*80)
        print(f"Testing: {transcript_file.stem}")
        print("="*80)
        
        # Load transcript
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
            text = transcript_data.get('text', '')
        
        print(f"Transcript length: {len(text)} characters")
        
        # Extract predictions using the new prompts
        print("\nExtracting predictions with new prompts...")
        try:
            predictions = extractor.extract_predictions(
                transcript_text=text,
                episode_info={
                    'title': transcript_file.stem,
                    'upload_date': transcript_data.get('upload_date', ''),
                    'video_id': transcript_file.stem.split('_')[-2] if '_' in transcript_file.stem else ''
                }
            )
            
            print(f"\nFound {len(predictions)} predictions!")
            
            # Display predictions
            for i, pred in enumerate(predictions):
                print(f"\nPrediction {i+1}:")
                print(f"  Asset: {pred.get('asset', 'N/A')}")
                print(f"  Price: {pred.get('price', 'N/A')}")
                print(f"  Date: {pred.get('date', 'N/A')}")
                print(f"  Confidence: {pred.get('confidence', 'N/A')}")
                print(f"  Speaker: {pred.get('speaker', 'N/A')}")
                print(f"  Context: {pred.get('context', '')[:200]}...")
            
            all_predictions.extend(predictions)
            
        except Exception as e:
            print(f"Error extracting predictions: {e}")
    
    # Save results
    if all_predictions:
        output_dir = Path("data/episodes/bitcoin_dive_bar_analysis/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"test_new_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'prompts_source': 'Rogue Trader analysis',
                'episodes_tested': len(episodes_to_test),
                'total_predictions': len(all_predictions),
                'predictions': all_predictions
            }, f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")
        print(f"Total predictions found: {len(all_predictions)}")
    else:
        print("\n\nNo predictions found")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()