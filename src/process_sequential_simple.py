#!/usr/bin/env python3
"""
Simple Sequential GPU Processing - Process one episode at a time
Using the TranscriptionRunner framework properly
"""
import sys
from pathlib import Path
import json
import os
from datetime import datetime
import time
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
from infrastructure.vast_ai.config import WHISPER_MODEL
from downloaders.youtube_channel_downloader import YouTubeChannelDownloader
from predictions.prediction_tracker.tracker_optimized import OptimizedCryptoPredictionTracker as CryptoPredictionTracker


def process_single_episode(
    runner: TranscriptionRunner,
    episode_file: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Process a single episode using TranscriptionRunner"""
    episode_id = episode_file.stem
    print(f"\n{'='*80}")
    print(f"Processing: {episode_id}")
    print(f"{'='*80}")
    
    # Set up debug logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"debug_{episode_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def debug_log(msg):
        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            f.flush()  # Ensure immediate write
    
    result = {
        "episode": episode_id,
        "file": str(episode_file),
        "status": "failed",
        "error": None,
        "instance_id": None,
        "predictions": []
    }
    
    try:
        # Create output directories
        transcripts_dir = output_dir / "transcripts"
        predictions_dir = output_dir / "predictions"
        transcripts_dir.mkdir(exist_ok=True)
        predictions_dir.mkdir(exist_ok=True)
        debug_log(f"Created output directories")
        
        # Use TranscriptionRunner to transcribe
        print("Transcribing with GPU...")
        debug_log(f"Starting transcription for {episode_file}")
        debug_log(f"File size: {episode_file.stat().st_size / 1024 / 1024:.2f} MB")
        transcript_result = runner.transcribe_audio(
            audio_path=episode_file,
            output_dir=transcripts_dir,
            model="base",
            use_faster_whisper=True
        )
        
        # Get instance ID from runner
        if runner.instance_manager.current_instance:
            result["instance_id"] = runner.instance_manager.current_instance.get('id')
            debug_log(f"Using instance ID: {result['instance_id']}")
        
        debug_log(f"Transcription completed in {transcript_result['metadata']['transcription_time']:.1f}s")
        debug_log(f"Transcript saved to: {transcripts_dir / f'{episode_file.stem}_transcript.json'}")
        
        # Extract predictions
        print("Extracting predictions...")
        debug_log("Starting prediction extraction")
        transcript_file = transcripts_dir / f"{episode_file.stem}_transcript.json"
        
        # Get episode info
        # Extract video ID from filename (YouTube IDs are always 11 characters)
        video_id = None
        stem = episode_file.stem
        if len(stem) >= 11 and '_' in stem:
            # Extract the last 11 characters after the last occurrence of '_' 
            # that leaves at least 11 characters
            for i in range(len(stem) - 11, -1, -1):
                if stem[i] == '_' and len(stem) - i - 1 >= 11:
                    potential_id = stem[i+1:i+12]
                    # Validate it looks like a YouTube ID (11 chars, alphanumeric + _ and -)
                    if len(potential_id) == 11 and all(c.isalnum() or c in '_-' for c in potential_id):
                        video_id = potential_id
                        break
        
        episode_info = {
            'title': episode_file.stem,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'id': episode_file.name,
            'video_id': video_id
        }
        
        # Process with tracker
        tracker = CryptoPredictionTracker()
        try:
            predictions_list = tracker.process_episode(str(transcript_file), episode_info)
        except Exception as e:
            debug_log(f"Error in tracker.process_episode: {e}")
            debug_log(f"Error type: {type(e).__name__}")
            import traceback
            debug_log(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Convert to dict format
        predictions_data = {
            'episode': episode_id,
            'predictions': []
        }
        
        # Debug and convert predictions
        for i, p in enumerate(predictions_list or []):
            try:
                debug_log(f"Processing prediction {i}: asset={p.asset}, value={p.value} (type={type(p.value)})")
                
                # Ensure value is a float
                value = float(p.value) if isinstance(p.value, (int, str)) else p.value
                
                pred_dict = {
                    'text': f"{p.asset} to ${value:,.0f}",
                    'asset': p.asset,
                    'value': value,
                    'confidence': p.confidence.value if hasattr(p.confidence, 'value') else str(p.confidence),
                    'timestamp': p.timestamp,
                    'timestamp_start': p.timestamp_start,
                    'timestamp_end': p.timestamp_end,
                    'youtube_link': p.youtube_link,
                    'context': p.raw_text[:200] if p.raw_text else '',
                    'timeframe': p.time_frame.value if p.time_frame and hasattr(p.time_frame, 'value') else str(p.time_frame)
                }
                predictions_data['predictions'].append(pred_dict)
            except Exception as e:
                debug_log(f"Error processing prediction {i}: {e}")
                debug_log(f"Prediction data: {p}")
                continue
        
        # Save predictions
        predictions_file = predictions_dir / f"{episode_id}_predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        print(f"Found {len(predictions_data['predictions'])} predictions")
        
        # Success!
        result["status"] = "success"
        result["predictions"] = predictions_data['predictions']
        result["transcription_time"] = transcript_result['metadata']['transcription_time']
        result["cost"] = transcript_result['metadata']['instance_cost']
        
        debug_log(f"Episode processed successfully!")
        debug_log(f"Total time: {result['transcription_time']:.1f}s")
        debug_log(f"Cost: ${result['cost']:.4f}")
        debug_log(f"Predictions found: {len(result['predictions'])}")
        
        print("\n✓ Episode processed successfully!")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"\n✗ Failed: {e}")
        debug_log(f"Error: {e}")
    
    return result


def main():
    # Configuration
    channel_url = "https://www.youtube.com/@BitcoinDiveBar"
    output_dir = Path("data/episodes/bitcoin_dive_bar_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Get API key
    api_key = os.getenv("VAST_API_KEY")
    if not api_key:
        api_key_file = Path.home() / ".config/vastai/vast_api_key"
        if api_key_file.exists():
            api_key = api_key_file.read_text().strip()
    
    if not api_key:
        print("Error: No Vast.ai API key found")
        return
    
    # Get downloaded files
    print("Checking downloaded episodes...")
    downloader = YouTubeChannelDownloader(str(output_dir))
    audio_files = [Path(f) for f in downloader.get_downloaded_files() if Path(f).exists()]
    print(f"Found {len(audio_files)} episodes")
    
    if not audio_files:
        print("No episodes found to process")
        return
    
    # Initialize transcription runner
    runner = TranscriptionRunner(api_key)
    
    # Process each episode
    results = []
    
    # PROCESS EPISODES 2-7
    test_files = audio_files[1:7]  # Index 1-6 = Episodes 2-7
    print(f"\n*** Processing episodes 2-7 ***")
    print(f"*** Will process {len(test_files)} episodes ***\n")
    
    try:
        # Set up GPU instance once
        print("\n=== Setting up GPU instance ===")
        instance = runner.setup_instance(
            gpu_type="RTX 3080",
            max_price=0.30  # Increased from 0.25 to match your original
        )
        print(f"Instance {instance['id']} ready")
        print(f"SSH: {instance.get('ssh_host')}:{instance.get('ssh_port')}")
        
        # Create main debug log
        main_log_file = output_dir / "logs" / f"main_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        def main_debug_log(msg):
            with open(main_log_file, 'a') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
                f.flush()
        
        main_debug_log(f"GPU instance {instance['id']} started")
        main_debug_log(f"Instance details: {json.dumps(instance, indent=2)}")
        
        for i, audio_file in enumerate(test_files):
            print(f"\n\nProcessing episode {i+1} of {len(test_files)}")
            
            # Retry logic - up to 3 attempts per episode
            max_retries = 3
            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"\nRetrying episode (attempt {attempt+1} of {max_retries})...")
                    time.sleep(30)  # Wait before retry
                
                result = process_single_episode(runner, audio_file, output_dir)
                
                if result["status"] == "success":
                    break  # Success, move to next episode
                else:
                    print(f"\nAttempt {attempt+1} failed: {result['error']}")
                    
                    # Check if it's a critical error that should stop everything
                    error_msg = str(result.get('error', '')).lower()
                    if 'no gpu offers available' in error_msg:
                        print("\nCRITICAL: No GPU offers available. Stopping.")
                        sys.exit(1)
                    
                    if attempt == max_retries - 1:
                        print(f"\nFailed after {max_retries} attempts. Moving to next episode.")
            
            results.append(result)
            
            # Save results after each episode
            results_file = output_dir / "processing_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Wait between episodes to avoid rate limiting
            if i < len(test_files) - 1:
                print("\nWaiting 30s before next episode...")
                time.sleep(30)
    
    finally:
        # Always clean up GPU instance
        print("\nCleaning up GPU instance...")
        runner.cleanup(destroy_instance=True)
    
    # Summary
    print("\n\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    total_predictions = sum(len(r["predictions"]) for r in results)
    
    print(f"Episodes processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total predictions found: {total_predictions}")
    
    if failed_count > 0:
        print("\nFailed episodes:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['episode']}: {r['error']}")


if __name__ == "__main__":
    main()