"""
Main prediction tracker for crypto podcasts
"""
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

from .models import Prediction, Outcome, Confidence, TimeFrame
from .llm_extractor import LLMPredictionExtractor
from .storage import PredictionStorage
from .analyzer import PredictionAnalyzer

class CryptoPredictionTracker:
    def __init__(self, data_dir: str = "prediction_data", api_key: Optional[str] = None):
        self.storage = PredictionStorage(data_dir)
        
        # Initialize LLM extractor (required)
        self.extractor = LLMPredictionExtractor(api_key)
        self.analyzer = PredictionAnalyzer(self.storage)
    
    def process_episode(self, transcript_file: str, episode_info: Dict) -> List[Prediction]:
        """Process a single episode transcript"""
        # Check if already processed
        episode_id = episode_info.get('id', transcript_file)
        if self.storage.is_episode_processed(episode_id):
            print(f"Episode {episode_id} already processed")
            return []
        
        # Read transcript
        if transcript_file.endswith('.json'):
            with open(transcript_file, 'r') as f:
                data = json.load(f)
                # Use new method that handles timestamps
                predictions = self.extractor.extract_predictions_with_timestamps(data, episode_info)
        else:
            # Fallback for non-JSON transcripts
            with open(transcript_file, 'r') as f:
                text = f.read()
            predictions = self.extractor.extract_predictions(text, episode_info)
        
        # Save predictions
        if predictions:
            self.storage.save_predictions(predictions)
            print(f"Found {len(predictions)} predictions in {episode_info.get('title', transcript_file)}")
            
            # Mark episode as processed
            self.storage.mark_episode_processed(episode_id, episode_info)
            
            # Don't update speaker stats here - that's for analysis phase
            # self.storage.update_speaker_stats()
        
        return predictions
    
    def record_outcome(self, asset: str, actual_value: float, 
                      date: Optional[str] = None) -> List[Outcome]:
        """Record actual outcome and check predictions"""
        if not date:
            date = datetime.now().isoformat()
        
        # Find relevant predictions
        predictions = self.storage.get_predictions_by_asset(asset)
        pending = [p for p in predictions if not self._has_outcome(p)]
        
        outcomes = []
        for pred in pending:
            # Check if prediction date has passed
            if self._should_check_prediction(pred, date):
                # Calculate accuracy
                was_correct, accuracy = self._evaluate_prediction(pred, actual_value)
                
                outcome = Outcome(
                    prediction_id=pred.prediction_id,
                    actual_value=actual_value,
                    actual_date=date,
                    was_correct=was_correct,
                    accuracy_score=accuracy,
                    notes=f"Target: ${pred.value}, Actual: ${actual_value}"
                )
                
                self.storage.save_outcome(outcome)
                outcomes.append(outcome)
                
                print(f"Recorded outcome for {asset}: "
                      f"Predicted ${pred.value}, Actual ${actual_value} "
                      f"({'CORRECT' if was_correct else 'INCORRECT'})")
        
        # Update speaker stats
        self.storage.update_speaker_stats()
        
        return outcomes
    
    def check_due_predictions(self) -> List[Prediction]:
        """Check which predictions are due for evaluation"""
        predictions = self.storage.get_pending_predictions()
        today = datetime.now()
        due = []
        
        for pred in predictions:
            if self._is_prediction_due(pred, today):
                due.append(pred)
        
        return due
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive report"""
        report = self.analyzer.generate_full_report()
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def _has_outcome(self, prediction: Prediction) -> bool:
        """Check if prediction already has an outcome"""
        outcomes = self.storage.load_outcomes()
        return any(o.prediction_id == prediction.prediction_id for o in outcomes)
    
    def _should_check_prediction(self, pred: Prediction, current_date: str) -> bool:
        """Determine if prediction should be checked"""
        # If specific date, check if passed
        if pred.predicted_date:
            return current_date >= pred.predicted_date
        
        # If timeframe, calculate from episode date
        if pred.time_frame and pred.episode_date:
            episode_date = datetime.fromisoformat(pred.episode_date)
            current = datetime.fromisoformat(current_date)
            
            if pred.time_frame == TimeFrame.DAYS and pred.time_value:
                target_date = episode_date + timedelta(days=pred.time_value)
            elif pred.time_frame == TimeFrame.WEEKS and pred.time_value:
                target_date = episode_date + timedelta(weeks=pred.time_value)
            elif pred.time_frame == TimeFrame.MONTHS and pred.time_value:
                target_date = episode_date + timedelta(days=pred.time_value * 30)
            elif pred.time_frame == TimeFrame.EOY:
                target_date = datetime(episode_date.year, 12, 31)
            else:
                return False
            
            return current >= target_date
        
        return False
    
    def _evaluate_prediction(self, pred: Prediction, actual_value: float) -> tuple[bool, float]:
        """Evaluate if prediction was correct"""
        # Define accuracy thresholds
        threshold = 0.1  # 10% margin
        
        # Ensure values are floats
        try:
            pred_value = float(pred.value)
            actual_val = float(actual_value)
        except (TypeError, ValueError) as e:
            print(f"Error converting values to float: pred.value={pred.value} (type={type(pred.value)}), actual_value={actual_value} (type={type(actual_value)})")
            raise ValueError(f"Invalid value types: {e}")
        
        # Calculate percentage difference
        if pred_value == 0:
            diff = 1.0  # Avoid division by zero
        else:
            diff = abs(pred_value - actual_val) / pred_value
        accuracy = 1.0 - diff
        
        # Consider correct if within threshold
        was_correct = diff <= threshold
        
        # Adjust for confidence level
        if pred.confidence == Confidence.LOW and diff <= 0.2:
            was_correct = True  # More lenient for low confidence
        elif pred.confidence == Confidence.HIGH and diff > 0.05:
            was_correct = False  # Stricter for high confidence
        
        return was_correct, accuracy
    
    def _is_prediction_due(self, pred: Prediction, current_date: datetime) -> bool:
        """Check if prediction is due for checking"""
        # Similar to _should_check_prediction but with datetime
        return self._should_check_prediction(pred, current_date.isoformat())

# CLI Interface
def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Prediction Tracker")
    parser.add_argument('command', choices=['process', 'outcome', 'report', 'check'],
                       help='Command to run')
    parser.add_argument('--file', help='Transcript file to process')
    parser.add_argument('--episode', help='Episode title')
    parser.add_argument('--date', help='Episode date (YYYY-MM-DD)')
    parser.add_argument('--asset', help='Asset symbol (BTC, ETH, etc.)')
    parser.add_argument('--value', type=float, help='Actual value')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    tracker = CryptoPredictionTracker()
    
    if args.command == 'process':
        if not args.file:
            print("Error: --file required for process command")
            return
        
        episode_info = {
            'title': args.episode or os.path.basename(args.file),
            'date': args.date or datetime.now().strftime('%Y-%m-%d'),
            'id': os.path.basename(args.file)
        }
        
        predictions = tracker.process_episode(args.file, episode_info)
        
        # Show predictions found
        for pred in predictions:
            print(f"\n{pred.asset}: ${pred.value:,.0f}")
            if pred.predicted_date:
                print(f"  By: {pred.predicted_date}")
            elif pred.time_frame:
                print(f"  In: {pred.time_value} {pred.time_frame.value}")
            print(f"  Confidence: {pred.confidence.value}")
            print(f"  Quote: {pred.raw_text}")
    
    elif args.command == 'outcome':
        if not args.asset or args.value is None:
            print("Error: --asset and --value required for outcome command")
            return
        
        outcomes = tracker.record_outcome(args.asset, args.value, args.date)
        print(f"Recorded {len(outcomes)} outcomes")
    
    elif args.command == 'report':
        report = tracker.generate_report(args.output)
        if not args.output:
            print(report)
    
    elif args.command == 'check':
        due = tracker.check_due_predictions()
        print(f"\n{len(due)} predictions are due for checking:\n")
        for pred in due:
            print(f"{pred.asset}: ${pred.value:,.0f} - {pred.raw_text[:50]}...")

if __name__ == "__main__":
    main()