#!/usr/bin/env python3
"""
Check all predictions and their current status
"""
import json
from datetime import datetime

# Load all processing results
with open('data/episodes/bitcoin_dive_bar_analysis/processing_results.json', 'r') as f:
    results = json.load(f)

# Get current date for reference
today = datetime.now().strftime('%Y-%m-%d')
print(f'Analysis Date: {today}')
print('='*80)

# Collect all predictions
all_predictions = []
for episode in results:
    if 'predictions' in episode:
        ep_name = episode['episode'].split(' - ')[0]  # Get "Bitcoin Dive Bar EP X"
        for pred in episode['predictions']:
            pred['episode'] = ep_name
            all_predictions.append(pred)

# Sort by asset and value
all_predictions.sort(key=lambda x: (x['asset'], x['value']))

# Group by asset
print("\nALL PREDICTIONS FOUND:\n")
for asset in ['BTC', 'SMLR', 'MSTR']:
    preds = [p for p in all_predictions if p['asset'] == asset]
    if preds:
        print(f'{asset} Predictions:')
        print('-' * 40)
        for p in preds:
            timeframe = p.get('timeframe', 'unspecified')
            if timeframe == 'None' or not timeframe:
                timeframe = 'unspecified'
            
            print(f'\n  Target: ${p["value"]:,.0f}')
            print(f'  Episode: {p["episode"]}')
            print(f'  Confidence: {p["confidence"]}')
            print(f'  Timeframe: {timeframe}')
            print(f'  Context: "{p["context"][:100]}..."')
            print(f'  Timestamp: {p["timestamp"]}')
            print(f'  Watch: {p["youtube_link"]}')

print('\n' + '='*80)
print('\nSUMMARY:')
print(f'Total predictions: {len(all_predictions)}')
print(f'BTC predictions: {len([p for p in all_predictions if p["asset"] == "BTC"])}')
print(f'SMLR predictions: {len([p for p in all_predictions if p["asset"] == "SMLR"])}')
print(f'MSTR predictions: {len([p for p in all_predictions if p["asset"] == "MSTR"])}')

# Check for near-term predictions
print('\n' + '='*80)
print('\nNEAR-TERM PREDICTIONS TO WATCH:')
for p in all_predictions:
    timeframe = p.get('timeframe', '').lower()
    context = p.get('context', '').lower()
    
    # Check for near-term indicators
    if any(term in context for term in ['next week', 'tomorrow', 'today', 'this week']):
        print(f'\n{p["asset"]} to ${p["value"]:,.0f}')
        print(f'  Context: "{p["context"]}"')
        print(f'  Episode: {p["episode"]}')