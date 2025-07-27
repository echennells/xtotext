#!/usr/bin/env python3
"""
Show predictions grouped by run_id for development tracking
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import calendar

def convert_timeframe_to_concrete(timeframe, time_value, episode_date, predicted_date=None):
    """Convert relative timeframes to concrete dates based on episode date"""
    if not episode_date:
        return timeframe
    
    try:
        # Parse episode date
        ep_date = datetime.strptime(episode_date, '%Y-%m-%d')
        
        # If we already have a predicted_date, use it
        if predicted_date:
            return predicted_date
        
        # Convert based on timeframe type
        if timeframe == 'days':
            days = time_value if time_value is not None else 1
            target_date = ep_date + timedelta(days=days)
            return f"{target_date.strftime('%B %d, %Y')} (~{days} days from episode)"
            
        elif timeframe == 'weeks':
            weeks = time_value if time_value is not None else 1
            target_date = ep_date + timedelta(weeks=weeks)
            # Determine which week of the month
            week_of_month = (target_date.day - 1) // 7 + 1
            ordinal = ['1st', '2nd', '3rd', '4th', '5th'][min(week_of_month - 1, 4)]
            return f"{ordinal} week of {target_date.strftime('%B %Y')} (~{weeks} week{'s' if weeks > 1 else ''} from episode)"
            
        elif timeframe == 'months':
            months = time_value if time_value is not None else 1
            # Add months (approximate)
            target_month = ep_date.month + months
            target_year = ep_date.year
            while target_month > 12:
                target_month -= 12
                target_year += 1
            return f"{calendar.month_name[target_month]} {target_year} (~{months} month{'s' if months > 1 else ''} from episode)"
            
        elif timeframe == 'end_of_year':
            return f"End of {ep_date.year}"
            
        elif timeframe == 'specific_date' and predicted_date:
            pred_date = datetime.strptime(predicted_date, '%Y-%m-%d')
            return pred_date.strftime('%B %d, %Y')
            
        elif timeframe == 'None' or not timeframe:
            return 'No timeframe specified'
            
        else:
            # For other timeframes, just return as is
            return timeframe
            
    except Exception as e:
        # If parsing fails, return original timeframe
        return timeframe

# Load predictions
pred_file = Path('data/episodes/bitcoin_dive_bar_analysis/prediction_data/latest/predictions.json')
if not pred_file.exists():
    print(f"No predictions file found at {pred_file}")
    exit(1)

with open(pred_file) as f:
    data = json.load(f)

# Check if it's the new format
if isinstance(data, dict) and 'all_predictions' in data:
    predictions = data['all_predictions']
    print(f"Total predictions across all runs: {len(predictions)}")
    print(f"Runs: {', '.join(data.get('runs', []))}")
    print("="*80)
else:
    # Old format
    predictions = data if isinstance(data, list) else []
    print(f"Old format - {len(predictions)} predictions")

# Group by run_id
by_run = defaultdict(list)
for pred in predictions:
    run_id = pred.get('run_id', 'no_run_id')
    by_run[run_id].append(pred)

# Display by run
for run_id in sorted(by_run.keys()):
    run_preds = by_run[run_id]
    print(f"\nRun: {run_id}")
    print(f"Predictions: {len(run_preds)}")
    
    # Show which model was used for this run
    models_used = set(p.get('model_used', 'unknown') for p in run_preds)
    if models_used:
        print(f"Model(s): {', '.join(models_used)}")
    
    print("-"*80)
    
    # Group by podcast series first, then episode
    by_series = defaultdict(lambda: defaultdict(list))
    for pred in run_preds:
        series = pred.get('podcast_series', 'Unknown Podcast')
        episode = pred.get('episode', 'Unknown Episode')
        by_series[series][episode].append(pred)
    
    for series, episodes in sorted(by_series.items()):
        print(f"\n  📻 {series}:")
        for episode, episode_preds in sorted(episodes.items()):
            # Shorten episode name by removing series prefix if present
            display_episode = episode
            if episode.startswith(series):
                display_episode = episode[len(series):].strip(' -')
            print(f"\n    {display_episode} ({len(episode_preds)} predictions):")
            
            for i, pred in enumerate(episode_preds, 1):
                asset = pred.get('asset', 'Unknown')
                value = pred.get('value', 0)
                confidence = pred.get('confidence', 'unknown')
                
                # Get timeframe and convert to concrete date
                raw_timeframe = pred.get('timeframe')
                time_value = pred.get('time_value')
                episode_date = pred.get('episode_date')
                predicted_date = pred.get('predicted_date')
                
                # Convert relative timeframe to concrete date
                concrete_timeframe = convert_timeframe_to_concrete(
                    raw_timeframe, time_value, episode_date, predicted_date
                )
                
                # Format price
                if value >= 1000:
                    price_str = f'${value:,.0f}'
                else:
                    price_str = f'${value}'
                
                # Get YouTube link
                timestamp = pred.get('timestamp')
                episode_id = pred.get('episode_id')
                youtube_link = None
                
                # Check if we have a direct youtube_link field
                if pred.get('youtube_link'):
                    youtube_link = pred['youtube_link']
                # Otherwise, try to construct it
                elif episode_id and timestamp:
                    # Parse timestamp to seconds
                    try:
                        # Skip invalid timestamps
                        if timestamp in ['unknown', '00:00:00', 'None'] or not timestamp:
                            youtube_link = f"https://youtube.com/watch?v={episode_id}"
                        else:
                            parts = timestamp.split(':')
                            if len(parts) == 2:  # MM:SS
                                seconds = int(parts[0]) * 60 + int(parts[1])
                            elif len(parts) == 3:  # H:MM:SS
                                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                            else:
                                seconds = 0
                            
                            # Only add timestamp if it's greater than 0
                            if seconds > 0:
                                youtube_link = f"https://youtube.com/watch?v={episode_id}&t={seconds}s"
                            else:
                                youtube_link = f"https://youtube.com/watch?v={episode_id}"
                    except:
                        youtube_link = f"https://youtube.com/watch?v={episode_id}"
                
                print(f"      {i}. {asset} → {price_str}")
                print(f"         Confidence: {confidence}")
                print(f"         Timeframe: {concrete_timeframe}")
                if episode_date:
                    print(f"         Episode Date: {episode_date}")
                if youtube_link:
                    print(f"         YouTube: {youtube_link}")
                if pred.get('reasoning'):
                    print(f"         Reasoning: {pred['reasoning'][:100]}...")

print(f"\n{'='*80}")
print("SUMMARY:")
print(f"Total runs: {len(by_run)}")
print(f"Total predictions: {len(predictions)}")

# Show breakdown by podcast series
series_counts = defaultdict(int)
for pred in predictions:
    series = pred.get('podcast_series', 'Unknown Podcast')
    series_counts[series] += 1

print(f"\nBreakdown by podcast series:")
for series, count in sorted(series_counts.items()):
    print(f"  📻 {series}: {count} predictions")

# Show prediction changes across runs
print(f"\n{'='*80}")
print("PREDICTION CHANGES ACROSS RUNS:")

# Group by episode+asset+value to see how many times each prediction appears
prediction_counts = defaultdict(lambda: {'runs': []})
for pred in predictions:
    key = (pred.get('episode', ''), pred.get('asset', ''), pred.get('value', 0))
    run_id = pred.get('run_id', 'no_run_id')
    prediction_counts[key]['runs'].append(run_id)

# Show predictions that appear in multiple runs
for (episode, asset, value), info in sorted(prediction_counts.items()):
    if len(set(info['runs'])) > 1:
        price_str = f'${value:,.0f}' if value >= 1000 else f'${value}'
        print(f"\n{asset} → {price_str} (Episode: {episode[:50]}...)")
        print(f"  Appears in {len(set(info['runs']))} runs: {', '.join(sorted(set(info['runs'])))}")