#!/usr/bin/env python3
"""
Debug script to test prediction extraction step by step
"""
import sys
import json
from pathlib import Path
from decimal import Decimal

# Add paths for imports
sys.path.insert(0, '/workspace/xtotext')
sys.path.insert(0, '/workspace/xtotext/src')
sys.path.insert(0, '/workspace/xtotext/config')

from predictions.prediction_tracker.tracker_optimized import OptimizedCryptoPredictionTracker


def decimal_to_float(obj):
    """Convert Decimal to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main():
    # Load transcript
    transcript_path = Path('/workspace/transcript.json')
    print(f"Loading transcript from: {transcript_path}")
    
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Create episode info
    episode_info = {
        'title': transcript_data.get('title', 'Unknown Episode'),
        'date': transcript_data.get('date', '2024-01-01'),
        'id': transcript_data.get('id', 'unknown-id'),
        'video_id': transcript_data.get('video_id')
    }
    
    print(f"\nEpisode info: {episode_info}")
    
    # Create tracker
    print("\nInitializing tracker...")
    tracker = OptimizedCryptoPredictionTracker()
    
    # Extract predictions WITHOUT saving
    print("\nExtracting predictions (without saving)...")
    predictions = tracker.extractor.extract_predictions_from_file(str(transcript_path), episode_info)
    
    print(f"\nFound {len(predictions)} predictions")
    
    # Debug first prediction
    if predictions:
        p = predictions[0]
        print(f"\nFirst prediction details:")
        print(f"  Asset: {p.asset} (type: {type(p.asset)})")
        print(f"  Value: {p.value} (type: {type(p.value)})")
        print(f"  Confidence: {p.confidence} (type: {type(p.confidence)})")
        print(f"  Time frame: {p.time_frame} (type: {type(p.time_frame)})")
        print(f"  Raw text: {p.raw_text[:100]}...")
        
        # Try to convert to dict
        print("\nTrying to convert to dict...")
        try:
            pred_dict = {
                'text': f"{p.asset} to ${float(p.value):,.0f}",
                'asset': p.asset,
                'value': float(p.value),  # Force conversion to float
                'confidence': p.confidence.value if hasattr(p.confidence, 'value') else str(p.confidence),
                'timestamp': p.timestamp,
                'context': p.raw_text[:200] if p.raw_text else '',
                'timeframe': p.time_frame.value if p.time_frame and hasattr(p.time_frame, 'value') else str(p.time_frame)
            }
            print("Success! Dict created:")
            print(json.dumps(pred_dict, indent=2))
        except Exception as e:
            print(f"Error converting to dict: {e}")
            import traceback
            traceback.print_exc()
    
    # Try saving with custom encoder
    print("\nTrying to save with custom JSON encoder...")
    output_path = Path('/workspace/debug_predictions.json')
    
    results = {
        'episode': episode_info['title'],
        'predictions': []
    }
    
    for p in predictions:
        try:
            pred_dict = {
                'text': f"{p.asset} to ${float(p.value):,.0f}",
                'asset': p.asset,
                'value': float(p.value),
                'confidence': p.confidence.value if hasattr(p.confidence, 'value') else str(p.confidence),
                'timestamp': p.timestamp,
                'context': p.raw_text[:200] if p.raw_text else '',
                'timeframe': p.time_frame.value if p.time_frame and hasattr(p.time_frame, 'value') else str(p.time_frame)
            }
            results['predictions'].append(pred_dict)
        except Exception as e:
            print(f"Error processing prediction: {e}")
    
    # Save with custom encoder
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=decimal_to_float)
    
    print(f"\nSaved {len(results['predictions'])} predictions to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())