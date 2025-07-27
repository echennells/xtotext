#!/usr/bin/env python3
import json
from pathlib import Path

# Load predictions
pred_file = Path('data/episodes/bitcoin_dive_bar_analysis/prediction_data/latest/predictions.json')
with open(pred_file) as f:
    predictions = json.load(f)

# Sort by episode and timestamp
predictions.sort(key=lambda x: (x.get('episode', ''), x.get('timestamp_start', 0) or 0))

# Group by episode
current_episode = None
episode_count = 0
total_count = 0

for pred in predictions:
    episode = pred.get('episode', 'Unknown')
    if episode != current_episode:
        if current_episode:
            print(f'  Total predictions: {episode_count}')
        print(f'\n{"="*80}')
        print(f'{episode}')
        print(f'{"="*80}')
        current_episode = episode
        episode_count = 0
    
    episode_count += 1
    total_count += 1
    
    # Extract all fields
    asset = pred.get('asset', 'Unknown')
    value = pred.get('value', 0)
    confidence = pred.get('confidence', 'unknown')
    timeframe = pred.get('time_frame', 'Not specified')
    predicted_date = pred.get('predicted_date', 'Not specified')
    conditions = pred.get('conditions', 'None')
    speaker = pred.get('speaker', 'Unknown')
    timestamp = pred.get('timestamp', 'No timestamp')
    youtube_link = pred.get('youtube_link', 'No link')
    raw_text = pred.get('raw_text', '')
    reasoning = pred.get('reasoning', '')
    
    # Format price
    if value >= 1000:
        price_str = f'${value:,.0f}'
    else:
        price_str = f'${value}'
    
    print(f'\n{total_count}. {asset} → {price_str}')
    print(f'   Speaker: {speaker}')
    print(f'   Confidence: {confidence.upper()}')
    print(f'   Timeframe: {timeframe}')
    print(f'   Predicted Date: {predicted_date}')
    print(f'   Conditions: {conditions}')
    print(f'   YouTube: {youtube_link}')
    print(f'   Raw Quote: "{raw_text}"')
    if reasoning:
        print(f'   Reasoning: {reasoning}')

if current_episode:
    print(f'  Total predictions: {episode_count}')

print(f'\n{"="*80}')
print(f'SUMMARY: {total_count} total predictions across {len(set(p.get("episode", "") for p in predictions))} episodes')