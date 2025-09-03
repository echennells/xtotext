"""
Transcription workflow module - handles the full pipeline of downloading, transcribing, and post-processing
"""
import os
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import List, Dict, Optional, Any

from downloaders.youtube_channel_downloader import YouTubeChannelDownloader
from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
from processors.claude_transcript_postprocessor import postprocess_transcript_claude
from database import get_database, log_video, log_postprocessing


class TranscriptionWorkflow:
    """Manages the complete transcription workflow"""
    
    def __init__(
        self,
        output_base_dir: Path = Path("data/episodes"),
        vast_api_key: Optional[str] = None
    ):
        """
        Initialize the workflow
        
        Args:
            output_base_dir: Base directory for all output
            vast_api_key: Vast.ai API key for GPU transcription
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Get Vast.ai API key
        self.vast_api_key = vast_api_key or os.getenv("VAST_API_KEY")
        if not self.vast_api_key:
            try:
                from config.config import VAST_API_KEY
                self.vast_api_key = VAST_API_KEY
            except ImportError:
                pass
        
        # Initialize database
        self.db = get_database()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def process_channel(
        self,
        channel_url: str,
        channel_name: str = None,
        max_episodes: Optional[int] = None,
        season_filter: Optional[str] = None,
        skip_existing: bool = True,
        gpu_type: str = "RTX 3090",
        max_gpu_price: float = 0.30,
        whisper_model: str = "base",
        post_process: bool = True
    ) -> Dict[str, Any]:
        """
        Process a YouTube channel
        
        Args:
            channel_url: YouTube channel URL
            channel_name: Name for organizing files (extracted from URL if not provided)
            max_episodes: Maximum number of episodes to process
            season_filter: Filter for specific season (e.g., "S16" or "Season 16")
            skip_existing: Skip videos already in database
            gpu_type: GPU type for Vast.ai
            max_gpu_price: Maximum price per hour for GPU
            whisper_model: Whisper model to use
            post_process: Whether to post-process with Claude
            
        Returns:
            Dictionary with processing results
        """
        # Extract channel name from URL if not provided
        if not channel_name:
            channel_name = channel_url.split("@")[-1].lower()
        
        # Set up directories
        download_dir = self.output_base_dir / f"{channel_name}_downloads"
        analysis_dir = self.output_base_dir / f"{channel_name}_analysis"
        download_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing channel: {channel_url}")
        self.logger.info(f"Output directory: {analysis_dir}")
        
        # Step 1: Download episodes
        results = {
            'channel': channel_url,
            'episodes_processed': [],
            'episodes_skipped': [],
            'episodes_failed': [],
            'total_cost': 0.0
        }
        
        downloader = YouTubeChannelDownloader(str(download_dir))
        
        # Download episodes
        self.logger.info(f"Downloading from {channel_url}...")
        # Note: YouTubeChannelDownloader doesn't support title filtering directly
        # We'll download more and filter afterwards
        download_limit = max_episodes * 3 if max_episodes and season_filter else max_episodes
        download_result = downloader.download_channel(channel_url, max_videos=download_limit)
        
        # Get audio files
        audio_files = []
        if max_episodes:
            # If limiting episodes, get the most recent ones
            all_files = sorted(
                [Path(f) for f in downloader.get_downloaded_files() if Path(f).exists()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Filter by season if specified
            if season_filter:
                filtered_files = []
                season_terms = [season_filter.lower(), f"season {season_filter.lower().replace('s', '')}", f"s{season_filter.lower().replace('season ', '')}"]
                for f in all_files:
                    if any(term in f.stem.lower() for term in season_terms):
                        filtered_files.append(f)
                audio_files = filtered_files[:max_episodes]
            else:
                audio_files = all_files[:max_episodes]
        else:
            audio_files = [Path(f) for f in downloader.get_downloaded_files() if Path(f).exists()]
        
        self.logger.info(f"Found {len(audio_files)} episodes to process")
        
        # Initialize Vast.ai runner if we have the API key
        vast_runner = None
        if self.vast_api_key:
            vast_runner = TranscriptionRunner(self.vast_api_key)
        
        # Process each episode
        for i, audio_file in enumerate(audio_files):
            self.logger.info(f"\nProcessing episode {i+1}/{len(audio_files)}: {audio_file.name}")
            
            # Extract video ID
            video_id = self._extract_video_id(audio_file.stem)
            
            # Check if already processed
            if skip_existing and video_id:
                db_entry = self.db.get_video(f"youtube_{video_id}")
                if db_entry:
                    self.logger.info(f"Skipping - already in database: {db_entry.get('title', audio_file.name)}")
                    results['episodes_skipped'].append(audio_file.name)
                    continue
            
            # Set up paths
            transcripts_dir = analysis_dir / "transcripts"
            transcripts_dir.mkdir(exist_ok=True)
            transcript_file = transcripts_dir / f"{audio_file.stem}_transcript.json"
            
            try:
                # Step 2: Transcribe if needed
                if not transcript_file.exists():
                    if not vast_runner:
                        self.logger.warning("No Vast.ai API key - skipping transcription")
                        results['episodes_failed'].append({
                            'episode': audio_file.name,
                            'error': 'No Vast.ai API key'
                        })
                        continue
                    
                    self.logger.info("Starting GPU transcription...")
                    
                    # Set up GPU instance
                    instance = vast_runner.setup_instance(
                        gpu_type=gpu_type,
                        max_price=max_gpu_price,
                        max_retries=3
                    )
                    self.logger.info(f"GPU instance {instance['id']} ready")
                    
                    # Transcribe
                    transcript_result = vast_runner.transcribe_audio(
                        audio_path=audio_file,
                        output_dir=transcripts_dir,
                        model=whisper_model,
                        use_faster_whisper=False
                    )
                    
                    transcription_cost = instance.get('dph_total', 0.10) * (transcript_result['metadata']['transcription_time'] / 3600)
                    results['total_cost'] += transcription_cost
                    
                    self.logger.info(f"Transcription completed in {transcript_result['metadata']['transcription_time']:.1f}s")
                    
                    # Cleanup GPU
                    vast_runner.cleanup(destroy_instance=True)
                else:
                    self.logger.info("Transcript already exists, skipping GPU transcription")
                
                # Step 3: Post-process if requested
                postprocessed_file = None
                if post_process and transcript_file.exists():
                    self.logger.info("Post-processing transcript with Claude...")
                    try:
                        postprocessed_file = postprocess_transcript_claude(
                            transcript_path=transcript_file
                        )
                        self.logger.info("Post-processing completed")
                        results['total_cost'] += 0.02  # Approximate Claude cost
                    except Exception as e:
                        self.logger.warning(f"Post-processing failed: {e}")
                
                # Step 4: Log to database
                if transcript_file.exists() and video_id:
                    # Calculate stats
                    with open(transcript_file, 'r') as f:
                        transcript_data = json.load(f)
                    text = transcript_data.get('text', '')
                    if not text and 'segments' in transcript_data:
                        text = ' '.join(seg.get('text', '') for seg in transcript_data['segments'])
                    
                    # Log video
                    db_video_id = log_video(
                        transcript_path=transcript_file,
                        title=audio_file.stem.replace('_', ' '),
                        url=f"https://www.youtube.com/watch?v={video_id}" if video_id else None,
                        platform='youtube',
                        category='bitcoin',
                        source_script='transcription_workflow.py',
                        processing_stats={
                            'word_count': len(text.split()),
                            'char_count': len(text),
                            'vast_ai_cost': transcription_cost if 'transcription_cost' in locals() else 0.10,
                        },
                        metadata={
                            'channel': channel_name,
                            'channel_url': channel_url
                        }
                    )
                    
                    # Log post-processing
                    if postprocessed_file and postprocessed_file.exists():
                        log_postprocessing(
                            video_id=db_video_id,
                            postprocessed_path=postprocessed_file,
                            model='claude-3-haiku',
                            cost=0.02
                        )
                
                results['episodes_processed'].append(audio_file.name)
                
            except Exception as e:
                self.logger.error(f"Failed to process {audio_file.name}: {e}")
                results['episodes_failed'].append({
                    'episode': audio_file.name,
                    'error': str(e)
                })
            finally:
                # Ensure GPU is cleaned up
                if vast_runner and hasattr(vast_runner, 'instance_manager'):
                    if vast_runner.instance_manager.current_instance:
                        vast_runner.cleanup(destroy_instance=True)
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info("PROCESSING COMPLETE")
        self.logger.info(f"Processed: {len(results['episodes_processed'])}")
        self.logger.info(f"Skipped: {len(results['episodes_skipped'])}")
        self.logger.info(f"Failed: {len(results['episodes_failed'])}")
        self.logger.info(f"Total cost: ${results['total_cost']:.2f}")
        
        return results
    
    def _extract_video_id(self, filename: str) -> Optional[str]:
        """Extract YouTube video ID from filename"""
        # YouTube video IDs are always 11 characters
        if len(filename) >= 11:
            # Try to find an 11-character alphanumeric string
            for j in range(len(filename) - 10):
                potential_id = filename[j:j+11]
                if all(c.isalnum() or c in '_-' for c in potential_id):
                    return potential_id
        return None