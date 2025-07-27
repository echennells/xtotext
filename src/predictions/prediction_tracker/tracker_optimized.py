"""
Optimized Crypto Prediction Tracker with reduced memory footprint
"""
from typing import List, Dict, Optional
import json
import os
from pathlib import Path
from datetime import datetime
import gc
import time

from .llm_extractor_optimized import OptimizedLLMPredictionExtractor
from .models import Prediction
from .storage import PredictionStorage


class OptimizedCryptoPredictionTracker:
    """
    Memory-optimized tracker for crypto predictions from podcast transcripts
    """
    
    def __init__(self, storage_dir: str = "prediction_data"):
        self.storage = PredictionStorage(storage_dir)
        self.extractor = OptimizedLLMPredictionExtractor()
        
        # Memory management settings
        self.enable_gc = True
        self.gc_threshold = 100 * 1024 * 1024  # 100MB
        self.processing_delay = 0.1  # seconds between heavy operations
    
    def process_episode(self, transcript_file: str, episode_info: Dict) -> List[Prediction]:
        """
        Process a single episode transcript with memory optimization
        
        Args:
            transcript_file: Path to transcript JSON file
            episode_info: Episode metadata (title, date, etc.)
            
        Returns:
            List of extracted predictions
        """
        print(f"\n=== OPTIMIZED PROCESSING ===")
        print(f"Episode: {episode_info.get('title', 'Unknown')}")
        print(f"File size: {os.path.getsize(transcript_file) / 1024 / 1024:.2f} MB")
        
        # Check memory before processing
        self._check_memory_usage()
        
        try:
            # Use the optimized extraction that streams the file
            predictions = self.extractor.extract_predictions_from_file(
                transcript_file, 
                episode_info
            )
            
            if predictions:
                # Process in smaller batches to reduce memory spikes
                self._save_predictions_in_batches(predictions, episode_info)
            
            # Force garbage collection after processing
            if self.enable_gc:
                gc.collect()
                time.sleep(self.processing_delay)
            
            return predictions
            
        except Exception as e:
            print(f"Error processing episode: {e}")
            raise
    
    def _save_predictions_in_batches(self, predictions: List[Prediction], episode_info: Dict, batch_size: int = 10):
        """Save predictions in batches to reduce memory usage"""
        # Save episode data using the existing method
        self.storage.mark_episode_processed(
            episode_info.get('id', ''),
            episode_info
        )
        
        # Save predictions in batches
        for i in range(0, len(predictions), batch_size):
            batch = predictions[i:i + batch_size]
            self.storage.save_predictions(batch)
            
            # Small delay between batches
            if i + batch_size < len(predictions):
                time.sleep(0.05)
    
    def _check_memory_usage(self):
        """Check current memory usage and trigger GC if needed"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            print(f"Current memory usage: {memory_mb:.1f} MB")
            
            if memory_mb > self.gc_threshold / (1024 * 1024):
                print("Memory threshold exceeded, forcing garbage collection...")
                gc.collect()
                time.sleep(0.5)
                
                # Check memory again
                new_memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"Memory after GC: {new_memory_mb:.1f} MB (freed {memory_mb - new_memory_mb:.1f} MB)")
        except ImportError:
            # psutil not available, skip memory check
            pass
    
    def process_multiple_episodes(self, transcript_files: List[str], delay_between_episodes: float = 2.0):
        """
        Process multiple episodes with memory optimization
        
        Args:
            transcript_files: List of transcript file paths
            delay_between_episodes: Seconds to wait between episodes
        """
        total_predictions = []
        
        for i, transcript_file in enumerate(transcript_files):
            print(f"\n{'='*60}")
            print(f"Processing episode {i+1} of {len(transcript_files)}")
            print(f"{'='*60}")
            
            # Extract episode info from filename
            filename = Path(transcript_file).stem
            episode_info = {
                'title': filename,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'id': filename
            }
            
            try:
                predictions = self.process_episode(transcript_file, episode_info)
                total_predictions.extend(predictions)
                
                print(f"✓ Found {len(predictions)} predictions")
                
            except Exception as e:
                print(f"✗ Failed to process {filename}: {e}")
            
            # Delay between episodes to prevent IO spikes
            if i < len(transcript_files) - 1:
                print(f"\nWaiting {delay_between_episodes}s before next episode...")
                time.sleep(delay_between_episodes)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"Total predictions extracted: {len(total_predictions)}")
        print(f"{'='*60}")
        
        return total_predictions