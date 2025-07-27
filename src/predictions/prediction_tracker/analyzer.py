"""
Analysis tools for crypto predictions
"""
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime
import statistics
import json

from .models import Prediction, Outcome, Speaker
from .storage import PredictionStorage

class PredictionAnalyzer:
    def __init__(self, storage: PredictionStorage):
        self.storage = storage
    
    def analyze_accuracy_by_asset(self) -> Dict[str, Dict]:
        """Analyze prediction accuracy by asset"""
        predictions = self.storage.load_predictions()
        outcomes = self.storage.load_outcomes()
        
        # Create outcome lookup
        outcome_map = {o.prediction_id: o for o in outcomes}
        
        # Group by asset
        asset_stats = defaultdict(lambda: {
            'total': 0, 'evaluated': 0, 'correct': 0,
            'accuracy_scores': [], 'values': []
        })
        
        for pred in predictions:
            stats = asset_stats[pred.asset]
            stats['total'] += 1
            stats['values'].append(pred.value)
            
            if pred.prediction_id in outcome_map:
                outcome = outcome_map[pred.prediction_id]
                stats['evaluated'] += 1
                if outcome.was_correct:
                    stats['correct'] += 1
                if outcome.accuracy_score is not None:
                    stats['accuracy_scores'].append(outcome.accuracy_score)
        
        # Calculate final stats
        results = {}
        for asset, stats in asset_stats.items():
            results[asset] = {
                'total_predictions': stats['total'],
                'evaluated': stats['evaluated'],
                'pending': stats['total'] - stats['evaluated'],
                'accuracy_rate': stats['correct'] / stats['evaluated'] if stats['evaluated'] > 0 else 0,
                'avg_accuracy_score': statistics.mean(stats['accuracy_scores']) if stats['accuracy_scores'] else 0,
                'avg_predicted_value': statistics.mean(stats['values']) if stats['values'] else 0,
                'value_range': (min(stats['values']), max(stats['values'])) if stats['values'] else (0, 0)
            }
        
        return results
    
    def analyze_accuracy_by_timeframe(self) -> Dict[str, float]:
        """Analyze accuracy by prediction timeframe"""
        predictions = self.storage.load_predictions()
        outcomes = self.storage.load_outcomes()
        outcome_map = {o.prediction_id: o for o in outcomes}
        
        timeframe_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        for pred in predictions:
            if pred.prediction_id in outcome_map:
                outcome = outcome_map[pred.prediction_id]
                
                # Categorize timeframe
                if pred.time_frame:
                    category = pred.time_frame.value
                elif pred.time_value:
                    if pred.time_value <= 7:
                        category = "short_term"
                    elif pred.time_value <= 30:
                        category = "medium_term"
                    else:
                        category = "long_term"
                else:
                    category = "unspecified"
                
                timeframe_stats[category]['total'] += 1
                if outcome.was_correct:
                    timeframe_stats[category]['correct'] += 1
        
        # Calculate accuracy rates
        results = {}
        for timeframe, stats in timeframe_stats.items():
            if stats['total'] > 0:
                results[timeframe] = stats['correct'] / stats['total']
        
        return results
    
    def analyze_confidence_correlation(self) -> Dict[str, Dict]:
        """Analyze if confidence levels correlate with accuracy"""
        predictions = self.storage.load_predictions()
        outcomes = self.storage.load_outcomes()
        outcome_map = {o.prediction_id: o for o in outcomes}
        
        confidence_stats = defaultdict(lambda: {
            'total': 0, 'correct': 0, 'accuracy_scores': []
        })
        
        for pred in predictions:
            if pred.prediction_id in outcome_map:
                outcome = outcome_map[pred.prediction_id]
                stats = confidence_stats[pred.confidence.value]
                
                stats['total'] += 1
                if outcome.was_correct:
                    stats['correct'] += 1
                if outcome.accuracy_score is not None:
                    stats['accuracy_scores'].append(outcome.accuracy_score)
        
        # Calculate results
        results = {}
        for confidence, stats in confidence_stats.items():
            if stats['total'] > 0:
                results[confidence] = {
                    'accuracy_rate': stats['correct'] / stats['total'],
                    'avg_accuracy_score': statistics.mean(stats['accuracy_scores']) if stats['accuracy_scores'] else 0,
                    'sample_size': stats['total']
                }
        
        return results
    
    def get_top_performers(self, min_predictions: int = 5) -> List[Dict]:
        """Get speakers with best prediction accuracy"""
        with open(self.storage.speakers_file, 'r') as f:
            speakers_data = json.load(f)
        
        # Filter by minimum predictions
        qualified = [s for s in speakers_data 
                    if s['total_predictions'] >= min_predictions]
        
        # Sort by accuracy
        qualified.sort(key=lambda x: x['accuracy_rate'], reverse=True)
        
        return qualified[:10]  # Top 10
    
    def analyze_trends(self) -> Dict[str, List]:
        """Analyze prediction trends over time"""
        predictions = self.storage.load_predictions()
        
        # Group by month
        monthly_stats = defaultdict(lambda: {
            'predictions': [], 'assets': defaultdict(int)
        })
        
        for pred in predictions:
            if pred.episode_date:
                month_key = pred.episode_date[:7]  # YYYY-MM
                monthly_stats[month_key]['predictions'].append(pred)
                monthly_stats[month_key]['assets'][pred.asset] += 1
        
        # Calculate trends
        trends = {
            'monthly_volume': {},
            'popular_assets': {},
            'avg_values': {}
        }
        
        for month, data in sorted(monthly_stats.items()):
            trends['monthly_volume'][month] = len(data['predictions'])
            trends['popular_assets'][month] = dict(data['assets'])
            
            # Average predicted values by asset
            asset_values = defaultdict(list)
            for pred in data['predictions']:
                asset_values[pred.asset].append(pred.value)
            
            trends['avg_values'][month] = {
                asset: statistics.mean(values)
                for asset, values in asset_values.items()
            }
        
        return trends
    
    def generate_full_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=== CRYPTO PREDICTION TRACKER REPORT ===")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        # Overall statistics
        predictions = self.storage.load_predictions()
        outcomes = self.storage.load_outcomes()
        report.append(f"Total Predictions: {len(predictions)}")
        report.append(f"Evaluated: {len(outcomes)}")
        report.append(f"Pending: {len(predictions) - len(outcomes)}\n")
        
        # Accuracy by asset
        report.append("=== ACCURACY BY ASSET ===")
        asset_stats = self.analyze_accuracy_by_asset()
        for asset, stats in sorted(asset_stats.items(), 
                                  key=lambda x: x[1]['total_predictions'], 
                                  reverse=True):
            report.append(f"\n{asset}:")
            report.append(f"  Total Predictions: {stats['total_predictions']}")
            report.append(f"  Evaluated: {stats['evaluated']}")
            report.append(f"  Accuracy Rate: {stats['accuracy_rate']:.1%}")
            report.append(f"  Avg Predicted Value: ${stats['avg_predicted_value']:,.0f}")
            report.append(f"  Value Range: ${stats['value_range'][0]:,.0f} - ${stats['value_range'][1]:,.0f}")
        
        # Accuracy by timeframe
        report.append("\n=== ACCURACY BY TIMEFRAME ===")
        timeframe_stats = self.analyze_accuracy_by_timeframe()
        for timeframe, accuracy in sorted(timeframe_stats.items()):
            report.append(f"{timeframe}: {accuracy:.1%}")
        
        # Confidence correlation
        report.append("\n=== CONFIDENCE VS ACCURACY ===")
        confidence_stats = self.analyze_confidence_correlation()
        for confidence, stats in sorted(confidence_stats.items()):
            report.append(f"{confidence}: {stats['accuracy_rate']:.1%} "
                        f"(n={stats['sample_size']})")
        
        # Top performers
        report.append("\n=== TOP PERFORMERS ===")
        top_speakers = self.get_top_performers()
        for i, speaker in enumerate(top_speakers[:5], 1):
            report.append(f"{i}. {speaker['name']}: "
                        f"{speaker['accuracy_rate']:.1%} "
                        f"({speaker['correct_predictions']}/{speaker['total_predictions']})")
        
        # Recent predictions due
        report.append("\n=== PREDICTIONS DUE FOR EVALUATION ===")
        pending = self.storage.get_pending_predictions()
        due_soon = []
        for pred in pending:
            if pred.predicted_date and pred.predicted_date <= datetime.now().isoformat():
                due_soon.append(pred)
        
        for pred in due_soon[:10]:
            report.append(f"{pred.asset} ${pred.value:,.0f} - {pred.raw_text[:50]}...")
        
        return '\n'.join(report)