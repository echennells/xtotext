#!/usr/bin/env python3
"""
Batch Video Processor - Process multiple videos efficiently without destroying GPU instance between videos

This script processes a queue of videos, keeping the GPU instance alive throughout the entire batch.
Supports YouTube and X/Twitter videos, with optional time segments.

Usage:
    # Process from queue file
    python process_batch_videos.py --queue-file queue.txt

    # Process URLs directly
    python process_batch_videos.py --urls "url1" "url2" "url3"

    # Process with local whisper
    python process_batch_videos.py --queue-file queue.txt --local-transcribe

Queue file format (one per line):
    https://youtube.com/watch?v=VIDEO_ID
    https://x.com/user/status/1234567890
    https://youtube.com/watch?v=VIDEO_ID --start 2:20 --end 2:22
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
from dotenv import load_dotenv

load_dotenv(override=True)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
from infrastructure.local_whisper.runner import LocalWhisperRunner
from processors.claude_transcript_postprocessor import postprocess_transcript_claude
from database import get_database, log_video, log_postprocessing
from utils.url_utils import parse_video_url
from utils.logging_utils import setup_logging
from utils.config_utils import get_vast_api_key
from utils.transcript_utils import extract_text_from_transcript
from utils.filename_utils import sanitize_filename
from downloaders.audio_downloader import download_youtube_audio, download_x_audio


class VideoQueueItem:
    """Represents a single video in the processing queue"""

    def __init__(self, url: str, start_time: Optional[str] = None, end_time: Optional[str] = None):
        self.url = url
        self.start_time = start_time
        self.end_time = end_time
        self.platform = None
        self.video_id = None
        self.full_url = None
        self.status = 'pending'  # pending, processing, completed, failed, skipped
        self.error_message = None
        self.audio_path = None
        self.transcript_path = None

        # Extract video info
        self.platform, self.video_id, self.full_url = parse_video_url(self.url)

    def __str__(self):
        time_info = ""
        if self.start_time and self.end_time:
            time_info = f" [{self.start_time}-{self.end_time}]"
        return f"{self.platform}/{self.video_id}{time_info} ({self.status})"


class BatchVideoProcessor:
    """Processes multiple videos efficiently in batch"""

    def __init__(self, use_local_whisper: bool = False, force_reprocess: bool = False):
        self.use_local_whisper = use_local_whisper
        self.force_reprocess = force_reprocess
        self.gpu_runner = None
        self.local_runner = None
        self.db = get_database()
        self.logger = setup_logging('batch_video')

        # Track processing statistics
        self.stats = {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0,
            'gpu_time': 0,
            'download_time': 0
        }

    def load_queue_from_file(self, queue_file: Path) -> List[VideoQueueItem]:
        """Load video queue from a text file"""
        queue = []

        with open(queue_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse line for URL and optional time segments
                parts = line.split()
                url = parts[0]
                start_time = None
                end_time = None

                # Look for --start and --end flags
                for i, part in enumerate(parts):
                    if part == '--start' and i + 1 < len(parts):
                        start_time = parts[i + 1]
                    elif part == '--end' and i + 1 < len(parts):
                        end_time = parts[i + 1]

                item = VideoQueueItem(url, start_time, end_time)
                if item.platform and item.video_id:
                    queue.append(item)
                else:
                    self.logger.warning(f"Line {line_num}: Could not parse URL: {url}")

        return queue

    def load_queue_from_urls(self, urls: List[str]) -> List[VideoQueueItem]:
        """Load video queue from a list of URLs"""
        queue = []
        for url in urls:
            item = VideoQueueItem(url)
            if item.platform and item.video_id:
                queue.append(item)
            else:
                self.logger.warning(f"Could not parse URL: {url}")
        return queue

    def download_audio(self, item: VideoQueueItem, download_dir: Path) -> Optional[Path]:
        """Download audio for a video"""
        try:
            if item.platform == 'youtube':
                return download_youtube_audio(item.video_id, download_dir, item.start_time, item.end_time)
            elif item.platform == 'x':
                return download_x_audio(item.video_id, download_dir, item.full_url)
        except Exception as e:
            self.logger.error(f"Error downloading audio for {item}: {e}")
        return None

    def setup_gpu_instance(self):
        """Set up GPU instance for transcription"""
        if self.use_local_whisper:
            self.logger.info("Using local Whisper Docker container")
            self.local_runner = LocalWhisperRunner(model="base")
            if not self.local_runner.setup():
                raise RuntimeError("Failed to setup Docker environment")
        else:
            self.logger.info("Setting up Vast.ai GPU instance...")
            vast_api_key = get_vast_api_key()

            if not vast_api_key:
                raise ValueError("No Vast.ai API key found")

            self.gpu_runner = TranscriptionRunner(vast_api_key)
            instance = self.gpu_runner.setup_instance(
                gpu_type="RTX 3090",
                max_price=0.30,
                max_retries=3
            )
            self.logger.info(f"GPU instance {instance['id']} ready")

    def cleanup_gpu_instance(self):
        """Clean up GPU instance"""
        if self.gpu_runner:
            self.logger.info("Cleaning up Vast.ai GPU instance...")
            self.gpu_runner.cleanup(destroy_instance=True)
            self.gpu_runner = None

        if self.local_runner:
            self.local_runner = None

    def transcribe_audio(self, audio_path: Path, output_dir: Path) -> Optional[Path]:
        """Transcribe audio file"""
        sanitized_name = sanitize_filename(f"{audio_path.stem}_transcript.json")
        transcript_path = output_dir / sanitized_name

        # Check if already exists
        if transcript_path.exists() and not self.force_reprocess:
            self.logger.info(f"Transcript already exists: {transcript_path}")
            return transcript_path

        try:
            if self.use_local_whisper:
                result = self.local_runner.transcribe_file(
                    str(audio_path),
                    output_path=str(transcript_path),
                    output_format="json"
                )
                return transcript_path if transcript_path.exists() else None
            else:
                result = self.gpu_runner.transcribe_audio(
                    audio_path=audio_path,
                    output_dir=output_dir,
                    model="base",
                    use_faster_whisper=False
                )
                return transcript_path if transcript_path.exists() else None

        except Exception as e:
            self.logger.error(f"Error transcribing {audio_path}: {e}")
            return None

    def process_queue(self, queue: List[VideoQueueItem]) -> Dict[str, Any]:
        """Process a queue of videos"""
        self.stats['total'] = len(queue)
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Processing {len(queue)} videos")
        print(f"{'='*60}\n")

        # Set up GPU instance once for all videos
        try:
            self.setup_gpu_instance()
        except Exception as e:
            self.logger.error(f"Failed to setup GPU instance: {e}")
            return self.stats

        try:
            for idx, item in enumerate(queue, 1):
                print(f"\n[{idx}/{len(queue)}] Processing: {item.full_url}")

                # Check if already in database
                db_video_id = f"{item.platform}_{item.video_id}"
                if self.db.get_video(db_video_id) and not self.force_reprocess:
                    print(f"  ✓ Already in database, skipping")
                    item.status = 'skipped'
                    self.stats['skipped'] += 1
                    continue

                item.status = 'processing'
                process_start = time.time()

                try:
                    # Set up directories
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    if item.platform == 'youtube':
                        download_dir = Path(f"data/youtube_downloads/{timestamp}")
                        output_dir = Path("data/youtube_analysis/transcripts")
                    else:
                        download_dir = Path(f"data/x_downloads/{timestamp}")
                        output_dir = Path("data/x_analysis/transcripts")

                    download_dir.mkdir(parents=True, exist_ok=True)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Step 1: Download audio
                    print("  Downloading audio...")
                    download_start = time.time()
                    audio_path = self.download_audio(item, download_dir)
                    if not audio_path:
                        raise RuntimeError("Failed to download audio")
                    item.audio_path = audio_path
                    self.stats['download_time'] += time.time() - download_start
                    print(f"    ✓ Downloaded: {audio_path.name}")

                    # Step 2: Transcribe
                    print("  Transcribing...")
                    transcribe_start = time.time()
                    transcript_path = self.transcribe_audio(audio_path, output_dir)
                    if not transcript_path:
                        raise RuntimeError("Failed to transcribe audio")
                    item.transcript_path = transcript_path
                    self.stats['gpu_time'] += time.time() - transcribe_start
                    print(f"    ✓ Transcribed: {transcript_path.name}")

                    # Step 3: Post-process (optional)
                    postprocessed_path = None
                    try:
                        print("  Post-processing with Claude...")
                        postprocessed_path = postprocess_transcript_claude(transcript_path)
                        print(f"    ✓ Post-processed: {postprocessed_path.name}")
                    except Exception as e:
                        print(f"    ⚠ Post-processing failed: {e}")

                    # Step 4: Log to database
                    final_transcript = postprocessed_path if postprocessed_path else transcript_path
                    text = extract_text_from_transcript(final_transcript)

                    db_result = log_video(
                        transcript_path=final_transcript,
                        title=audio_path.stem.replace('_', ' '),
                        url=item.full_url,
                        platform=item.platform,
                        category='general',
                        source_script='process_batch_videos.py',
                        processing_stats={
                            'word_count': len(text.split()),
                            'char_count': len(text),
                            'processing_time': time.time() - process_start,
                            'batch_position': idx,
                            'batch_size': len(queue)
                        },
                        metadata={
                            'video_id': item.video_id,
                            'processed_date': datetime.now().isoformat(),
                            'batch_processing': True
                        }
                    )

                    print(f"  ✓ Completed: {db_result}")
                    item.status = 'completed'
                    self.stats['completed'] += 1

                except Exception as e:
                    self.logger.error(f"Error processing {item}: {e}")
                    print(f"  ✗ Failed: {e}")
                    item.status = 'failed'
                    item.error_message = str(e)
                    self.stats['failed'] += 1

                # Progress summary
                print(f"\nProgress: {self.stats['completed']} completed, "
                      f"{self.stats['failed']} failed, "
                      f"{self.stats['skipped']} skipped, "
                      f"{self.stats['total'] - idx} remaining")

        finally:
            # Clean up GPU instance after all videos are processed
            self.cleanup_gpu_instance()

        self.stats['total_time'] = time.time() - start_time

        # Final summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos: {self.stats['total']}")
        print(f"Completed: {self.stats['completed']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped: {self.stats['skipped']}")
        print(f"Total time: {self.stats['total_time']:.1f}s")
        print(f"GPU time: {self.stats['gpu_time']:.1f}s")
        print(f"Download time: {self.stats['download_time']:.1f}s")
        if self.stats['completed'] > 0:
            print(f"Avg time per video: {self.stats['total_time'] / self.stats['completed']:.1f}s")

        # Save queue status
        self._save_queue_status(queue)

        return self.stats

    def _save_queue_status(self, queue: List[VideoQueueItem]):
        """Save queue processing status to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        status_file = Path("logs") / f"batch_status_{timestamp}.json"

        status_data = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'queue': [
                {
                    'url': item.url,
                    'platform': item.platform,
                    'video_id': item.video_id,
                    'status': item.status,
                    'error': item.error_message,
                    'audio_path': str(item.audio_path) if item.audio_path else None,
                    'transcript_path': str(item.transcript_path) if item.transcript_path else None
                }
                for item in queue
            ]
        }

        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)

        print(f"\nQueue status saved to: {status_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple videos efficiently',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process from queue file
  python process_batch_videos.py --queue-file queue.txt

  # Process URLs directly
  python process_batch_videos.py --urls "https://youtube.com/watch?v=abc" "https://x.com/user/status/123"

  # Use local Whisper instead of Vast.ai
  python process_batch_videos.py --queue-file queue.txt --local-transcribe

  # Force reprocessing even if already in database
  python process_batch_videos.py --queue-file queue.txt --force

Queue file format (queue.txt):
  # Comments start with #
  https://youtube.com/watch?v=VIDEO_ID
  https://x.com/user/status/1234567890
  # YouTube with time segment
  https://youtube.com/watch?v=VIDEO_ID --start 2:20 --end 2:22
"""
    )

    parser.add_argument('--queue-file', type=Path, help='Path to queue file with URLs')
    parser.add_argument('--urls', nargs='+', help='URLs to process directly')
    parser.add_argument('--local-transcribe', action='store_true',
                       help='Use local Whisper Docker instead of Vast.ai')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force reprocessing even if already in database')

    args = parser.parse_args()

    if not args.queue_file and not args.urls:
        parser.error("Please provide either --queue-file or --urls")

    # Initialize processor
    processor = BatchVideoProcessor(
        use_local_whisper=args.local_transcribe,
        force_reprocess=args.force
    )

    # Load queue
    if args.queue_file:
        if not args.queue_file.exists():
            print(f"Error: Queue file not found: {args.queue_file}")
            sys.exit(1)
        queue = processor.load_queue_from_file(args.queue_file)
    else:
        queue = processor.load_queue_from_urls(args.urls)

    if not queue:
        print("No valid videos found in queue")
        sys.exit(1)

    # Process the queue
    try:
        stats = processor.process_queue(queue)
        sys.exit(0 if stats['failed'] == 0 else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        processor.cleanup_gpu_instance()
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        processor.cleanup_gpu_instance()
        sys.exit(1)


if __name__ == "__main__":
    main()
