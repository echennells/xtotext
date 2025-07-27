#!/usr/bin/env python3
"""View debug logs from two-stage extraction"""
import json
import sys
from pathlib import Path
from datetime import datetime

def view_debug_logs():
    debug_dir = Path("logs/debug")
    if not debug_dir.exists():
        print("No debug logs found. Run prediction extraction first.")
        return
    
    # Get all debug log files sorted by date
    log_files = sorted(debug_dir.glob("debug_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not log_files:
        print("No debug log files found.")
        return
    
    print(f"Found {len(log_files)} debug logs:\n")
    
    # Show list of logs
    for i, log_file in enumerate(log_files[:10]):  # Show latest 10
        print(f"{i+1}. {log_file.name}")
    
    if len(sys.argv) > 1:
        # View specific log
        try:
            idx = int(sys.argv[1]) - 1
            if 0 <= idx < len(log_files):
                view_log(log_files[idx])
            else:
                print(f"\nInvalid selection. Choose 1-{len(log_files)}")
        except ValueError:
            # Try to find by filename
            for log_file in log_files:
                if sys.argv[1] in str(log_file):
                    view_log(log_file)
                    return
            print(f"\nLog file not found: {sys.argv[1]}")
    else:
        # View latest
        print("\nViewing latest log (use 'python view_debug_logs.py N' to view specific log):\n")
        view_log(log_files[0])

def view_log(log_file):
    """Display detailed debug log"""
    print(f"\n{'='*80}")
    print(f"Debug Log: {log_file.name}")
    print(f"{'='*80}\n")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    # Episode info
    print(f"Episode: {data['episode']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Models: {data['models']['snippet']} → {data['models']['prediction']}\n")
    
    # Summary
    if 'summary' in data:
        print("SUMMARY:")
        print(f"  Total snippets extracted: {data['summary']['total_snippets']}")
        print(f"  Total predictions found: {data['summary']['total_predictions']}")
        print(f"  Total cost: ${data['summary']['cost']['total']:.4f}")
        print()
    
    # Snippets
    print(f"SNIPPETS EXTRACTED ({len(data['snippets'])}):")
    print("-" * 80)
    for snippet in data['snippets']:
        print(f"\nSnippet {snippet['index'] + 1}:")
        print(f"  Reason: {snippet['reason']}")
        print(f"  Tokens: {snippet['tokens']}")
        print(f"  Text: {snippet['text'][:200]}..." if len(snippet['text']) > 200 else f"  Text: {snippet['text']}")
    
    # Predictions by snippet
    print(f"\n\nPREDICTIONS BY SNIPPET:")
    print("-" * 80)
    for pred_data in data['predictions']:
        print(f"\nFrom Snippet {pred_data['snippet_index'] + 1} (found {pred_data['predictions_found']} predictions):")
        print(f"  Snippet: {pred_data['snippet_text'][:150]}...")
        
        if pred_data['raw_predictions']:
            for i, pred in enumerate(pred_data['raw_predictions']):
                print(f"\n  Prediction {i+1}:")
                print(f"    Asset: {pred.get('asset', 'N/A')}")
                print(f"    Price: {pred.get('price', 'N/A')}")
                print(f"    Timeframe: {pred.get('timeframe', 'N/A')}")
                print(f"    Confidence: {pred.get('confidence', 'N/A')}")
                print(f"    Quote: {pred.get('quote', 'N/A')[:100]}...")
    
    # Errors
    if data.get('errors'):
        print(f"\n\nERRORS ({len(data['errors'])}):")
        print("-" * 80)
        for error in data['errors']:
            print(f"\n  Stage: {error['stage']}")
            print(f"  Reason: {error['reason']}")
            if 'raw_prediction' in error:
                print(f"  Failed data: {error['raw_prediction']}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    view_debug_logs()