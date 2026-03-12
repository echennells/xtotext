#!/usr/bin/env python3
"""
Process all videos from a YouTube channel through the full pipeline.
Keeps a single Vast.ai GPU instance alive for the entire batch to avoid
re-provisioning overhead between videos.

Usage:
    python process_channel.py <channel_url>
    python process_channel.py <channel_url> --max-videos 10
    python process_channel.py <channel_url> --force
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging
import time
from dotenv import load_dotenv

load_dotenv(override=True)

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
from processors.claude_transcript_postprocessor import postprocess_transcript_claude
from database import get_database, log_video, log_postprocessing
from downloaders.youtube_channel_downloader import YouTubeChannelDownloader
from utils.logging_utils import setup_logging
from utils.config_utils import get_vast_api_key
from utils.transcript_utils import extract_text_from_transcript
from utils.filename_utils import sanitize_filename
from downloaders.audio_downloader import download_youtube_audio


def get_channel_videos(channel_url):
    """Get list of (video_id, title, url) from a YouTube channel using yt-dlp."""
    downloader = YouTubeChannelDownloader(output_dir="data/channel_downloads")
    videos = downloader._get_channel_videos(channel_url)
    return videos


def process_video_on_gpu(runner, video_id, title, video_url, transcripts_dir, db, force, logger):
    """
    Run the full pipeline for a single video using an already-running GPU instance.
    Returns True on success, False on failure.
    """
    full_url = f"https://www.youtube.com/watch?v={video_id}"

    # Check database
    db_video_id = f"youtube_{video_id}"
    if db.get_video(db_video_id) and not force:
        print(f"  ✓ Already in database, skipping")
        return 'skipped'

    try:
        # Step 1: Download audio
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        download_dir = Path(f"data/youtube_downloads/{timestamp}")
        download_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading audio...")
        audio_file = download_youtube_audio(video_id, download_dir)
        print(f"  ✓ Downloaded: {audio_file.name}")

        # Step 2: Transcribe (reuse existing GPU instance)
        sanitized_name = sanitize_filename(f"{audio_file.stem}_transcript.json")
        transcript_file = transcripts_dir / sanitized_name

        if transcript_file.exists() and not force:
            print(f"  ✓ Transcript already exists, skipping transcription")
        else:
            print(f"  Transcribing...")
            runner.transcribe_audio(
                audio_path=audio_file,
                output_dir=transcripts_dir,
                model="base",
                use_faster_whisper=False
            )
            if not transcript_file.exists():
                raise RuntimeError(f"Transcript not found after transcription: {transcript_file}")
            print(f"  ✓ Transcribed: {transcript_file.name}")

        # Step 3: Post-process
        sanitized_postprocessed_name = sanitize_filename(f"{audio_file.stem}_transcript_claude_postprocessed.json")
        postprocessed_file = transcripts_dir / sanitized_postprocessed_name

        if postprocessed_file.exists() and not force:
            print(f"  ✓ Post-processed transcript already exists")
        else:
            try:
                print(f"  Post-processing with Claude...")
                postprocessed_file = postprocess_transcript_claude(transcript_path=transcript_file)
                print(f"  ✓ Post-processed: {postprocessed_file.name}")
            except Exception as e:
                print(f"  ⚠ Post-processing failed: {e}")
                postprocessed_file = None

        # Step 4: Log to database
        final_transcript = postprocessed_file if postprocessed_file and postprocessed_file.exists() else transcript_file
        text = extract_text_from_transcript(final_transcript)

        db_result = log_video(
            transcript_path=final_transcript,
            title=title,
            url=full_url,
            platform='youtube',
            category='general',
            source_script='process_channel.py',
            processing_stats={
                'word_count': len(text.split()),
                'char_count': len(text),
            },
            metadata={
                'video_id': video_id,
                'processed_date': datetime.now().isoformat(),
                'channel_batch': True
            }
        )

        if postprocessed_file and postprocessed_file.exists():
            log_postprocessing(
                video_id=db_result,
                postprocessed_path=postprocessed_file,
                model='claude-3-haiku',
                cost=0.02
            )

        print(f"  ✓ Logged to database: {db_result}")
        return 'success'

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        print(f"  ✗ Failed: {e}")
        return 'failed'


def main():
    parser = argparse.ArgumentParser(description='Process all videos from a YouTube channel')
    parser.add_argument('channel_url', help='YouTube channel URL (e.g. https://www.youtube.com/@ChannelName)')
    parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process')
    parser.add_argument('--force', '-f', action='store_true', help='Reprocess videos already in database')
    args = parser.parse_args()

    logger = setup_logging('channel')

    # Get video list
    print(f"\nFetching video list from: {args.channel_url}")
    videos = get_channel_videos(args.channel_url)

    if not videos:
        print("No videos found in channel.")
        sys.exit(1)

    print(f"Found {len(videos)} videos")

    if args.max_videos:
        videos = videos[:args.max_videos]
        print(f"Limiting to {args.max_videos} videos")

    # Check which are already done
    db = get_database()
    to_process = []
    for video_id, title, url in videos:
        db_video_id = f"youtube_{video_id}"
        if db.get_video(db_video_id) and not args.force:
            print(f"  [skip] {title}")
        else:
            to_process.append((video_id, title, url))
            print(f"  [queue] {title}")

    if not to_process:
        print("\nAll videos already processed. Use --force to reprocess.")
        sys.exit(0)

    print(f"\n{len(to_process)} videos to process ({len(videos) - len(to_process)} already done)")

    # Set up output dirs
    transcripts_dir = Path("data/youtube_analysis/transcripts")
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Get API key
    vast_api_key = get_vast_api_key()
    if not vast_api_key:
        print("Error: No Vast.ai API key found. Set VAST_API_KEY in your .env file.")
        sys.exit(1)

    # Set up GPU instance once for all videos
    runner = TranscriptionRunner(vast_api_key)
    print(f"\n{'='*60}")
    print("Setting up Vast.ai GPU instance (once for entire batch)...")
    print(f"{'='*60}")

    try:
        instance = runner.setup_instance(gpu_type="RTX 3090", max_price=0.30, max_retries=3)
        print(f"✓ GPU instance {instance['id']} ready\n")
    except Exception as e:
        print(f"✗ Failed to set up GPU instance: {e}")
        sys.exit(1)

    # Process each video
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    failed_videos = []
    start_time = time.time()

    try:
        for idx, (video_id, title, url) in enumerate(to_process, 1):
            print(f"\n[{idx}/{len(to_process)}] {title}")
            print(f"  URL: https://www.youtube.com/watch?v={video_id}")

            result = process_video_on_gpu(
                runner=runner,
                video_id=video_id,
                title=title,
                video_url=url,
                transcripts_dir=transcripts_dir,
                db=db,
                force=args.force,
                logger=logger
            )

            stats[result] += 1
            if result == 'failed':
                failed_videos.append((video_id, title))

            elapsed = time.time() - start_time
            remaining = len(to_process) - idx
            avg_per_video = elapsed / idx
            eta = avg_per_video * remaining
            print(f"  Progress: {idx}/{len(to_process)} | "
                  f"✓ {stats['success']} ✗ {stats['failed']} skip {stats['skipped']} | "
                  f"ETA: {eta/60:.0f}m")

    finally:
        print(f"\n{'='*60}")
        print("Destroying GPU instance...")
        runner.cleanup(destroy_instance=True)
        print("✓ GPU instance destroyed")

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("CHANNEL PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos:  {len(to_process)}")
    print(f"Succeeded:     {stats['success']}")
    print(f"Failed:        {stats['failed']}")
    print(f"Skipped:       {stats['skipped']}")
    print(f"Total time:    {total_time/60:.1f}m")
    if stats['success'] > 0:
        print(f"Avg per video: {total_time/max(stats['success'],1)/60:.1f}m")

    if failed_videos:
        print(f"\nFailed videos:")
        for vid_id, vid_title in failed_videos:
            print(f"  - {vid_title} (https://www.youtube.com/watch?v={vid_id})")

    sys.exit(0 if stats['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
