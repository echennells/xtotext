import json
from datetime import datetime

# Load predictions
with open('prediction_data/predictions.json', 'r') as f:
    predictions = json.load(f)

# Analyze timeframes
timeframe_counts = {}
september_predictions = []
missing_timeframe = []

for pred in predictions:
    tf = pred.get('time_frame', 'None')
    tv = pred.get('time_value', 'None')
    key = f'{tf}_{tv}'
    timeframe_counts[key] = timeframe_counts.get(key, 0) + 1
    
    # Check for September predictions
    if pred.get('predicted_date') and '2025-09' in str(pred.get('predicted_date')):
        september_predictions.append({
            'asset': pred.get('asset'),
            'value': pred.get('value'),
            'raw_text': pred.get('raw_text', '')[:100],
            'time_frame': tf,
            'time_value': tv,
            'episode_date': pred.get('episode_date'),
            'timestamp': pred.get('timestamp')
        })
    
    # Check for missing timeframes
    if not tf or tf == 'None':
        missing_timeframe.append(pred)

print('Timeframe distribution:')
for k, v in sorted(timeframe_counts.items(), key=lambda x: x[1], reverse=True):
    print(f'  {k}: {v}')

print(f'\nTotal predictions: {len(predictions)}')
print(f'Total September 2025 predictions: {len(september_predictions)}')
print(f'Missing timeframe: {len(missing_timeframe)}')

print('\nFirst 10 September predictions:')
for i, pred in enumerate(september_predictions[:10]):
    print(f'\n{i+1}. {pred["asset"]} ${pred["value"]:,.0f} - {pred["time_frame"]}_{pred["time_value"]} - Episode: {pred["episode_date"]}')
    print(f'   Timestamp: {pred["timestamp"]}')
    print(f'   Quote: {pred["raw_text"]}...')

# Analyze patterns in quotes for september predictions
print('\n\nAnalyzing quote patterns for September predictions:')
quote_patterns = {}
for pred in september_predictions:
    quote = pred['raw_text'].lower()
    
    # Look for common patterns
    patterns = []
    if 'bull' in quote: patterns.append('bull_market')
    if 'cycle' in quote: patterns.append('cycle')
    if 'run' in quote: patterns.append('run')
    if 'when' in quote or 'if' in quote: patterns.append('conditional')
    if 'momentum' in quote: patterns.append('momentum')
    if not any(word in quote for word in ['tomorrow', 'next week', 'month', 'year', 'soon', 'august', 'september']):
        patterns.append('no_explicit_time')
    
    for p in patterns:
        quote_patterns[p] = quote_patterns.get(p, 0) + 1

print('\nQuote patterns in September predictions:')
for pattern, count in sorted(quote_patterns.items(), key=lambda x: x[1], reverse=True):
    print(f'  {pattern}: {count}')