#!/usr/bin/env python3
"""
Process a single video (YouTube or X/Twitter) through the pipeline:
1. Download the video/audio
2. Transcribe using Vast.ai GPU
3. Post-process transcript with Claude
4. Log to database

Usage:
    python process_single_video.py <url>
    python process_single_video.py --youtube <video_id_or_url>
    python process_single_video.py --x <tweet_id_or_url>
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file (override shell variables)
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
from downloaders.audio_downloader import download_youtube_audio, download_x_audio, download_direct_url
import subprocess


def format_time_for_ffmpeg(time_str):
    """Convert HH:MM format to FFmpeg's expected HH:MM:SS format"""
    # Simply append :00 for seconds
    return f"{time_str}:00"


def trim_audio_on_do(audio_file, start_time, end_time, output_dir):
    """
    Trim audio file using Digital Ocean droplet to offload processing

    Args:
        audio_file: Path to the audio file
        start_time: Start time in HH:MM format (e.g., "2:20" for 2 hours 20 minutes)
        end_time: End time in HH:MM format (e.g., "2:22" for 2 hours 22 minutes)
        output_dir: Directory for trimmed output

    Returns:
        Path to trimmed audio file
    """
    print(f"\n{'='*60}")
    print("TRIMMING AUDIO ON DIGITAL OCEAN")
    print(f"{'='*60}")

    # Format times for ffmpeg (HH:MM:SS)
    ffmpeg_start = format_time_for_ffmpeg(start_time)
    ffmpeg_end = format_time_for_ffmpeg(end_time)

    print(f"Input times: {start_time} to {end_time}")
    print(f"FFmpeg format: {ffmpeg_start} to {ffmpeg_end}")

    # Initialize DO runner (imported here to avoid CONTROL_TOWER_SECRET being required at startup)
    from infrastructure.digital_ocean.simple_runner import SimpleDigitalOceanRunner
    do_runner = SimpleDigitalOceanRunner()

    try:
        # Upload audio file to DO
        remote_audio = f"/workspace/audio/{audio_file.name}"
        print(f"Uploading {audio_file.name} to Digital Ocean...")
        do_runner.upload_file(str(audio_file), remote_audio)

        # Prepare output filename (use original time strings for filename)
        trimmed_name = f"trimmed_{audio_file.stem}_{start_time.replace(':', '')}_{end_time.replace(':', '')}{audio_file.suffix}"
        remote_trimmed = f"/workspace/audio/{trimmed_name}"

        # Run ffmpeg on DO (use formatted times)
        ffmpeg_cmd = f'ffmpeg -i "{remote_audio}" -ss {ffmpeg_start} -to {ffmpeg_end} -c copy "{remote_trimmed}" -y'
        print(f"Trimming audio on Digital Ocean...")
        result = do_runner.run_command(ffmpeg_cmd)

        if "error" in result.lower():
            raise RuntimeError(f"FFmpeg trim failed: {result}")

        # Download trimmed file back
        local_trimmed = output_dir / trimmed_name
        print(f"Downloading trimmed audio...")
        do_runner.download_file(remote_trimmed, str(local_trimmed))

        # Clean up remote files
        do_runner.run_command(f"rm -f {remote_audio} {remote_trimmed}")

        print(f"✓ Trimmed audio saved to: {local_trimmed}")
        print(f"✓ Trimmed duration: {start_time} to {end_time}")

        return local_trimmed

    finally:
        # ALWAYS clean up the DO droplet
        print("Cleaning up Digital Ocean droplet...")
        do_runner.cleanup(destroy_droplet=True)


def transcribe_audio(audio_path, output_dir, vast_api_key):
    """Transcribe audio using Vast.ai GPU"""
    print("\nTranscribing with Vast.ai GPU...")

    runner = TranscriptionRunner(vast_api_key)

    try:
        # Set up GPU instance
        print("Starting Vast.ai GPU instance...")
        instance = runner.setup_instance(
            gpu_type="RTX 3090",
            max_price=0.30,
            max_retries=3
        )
        print(f"GPU instance {instance['id']} ready")

        # Transcribe
        result = runner.transcribe_audio(
            audio_path=audio_path,
            output_dir=output_dir,
            model="base",
            use_faster_whisper=False
        )

        print(f"Transcription completed in {result['metadata']['transcription_time']:.1f}s")

        # The transcript is already saved to output_dir with sanitized filename
        sanitized_name = sanitize_filename(f"{audio_path.stem}_transcript.json")
        transcript_path = output_dir / sanitized_name
        return transcript_path

    finally:
        # Cleanup GPU
        print("Cleaning up Vast.ai GPU instance...")
        runner.cleanup(destroy_instance=True)



def main():
    parser = argparse.ArgumentParser(description='Process a single video through the full pipeline')
    parser.add_argument('url', nargs='?', help='Video URL (YouTube or X/Twitter)')
    parser.add_argument('--youtube', help='YouTube video URL or ID')
    parser.add_argument('--x', '--twitter', help='X/Twitter status URL or ID')
    parser.add_argument('--local-transcribe', action='store_true', help='Use local Whisper Docker container instead of Vast.ai')
    parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation prompts and force reprocessing')
    parser.add_argument('--start', help='Start time in HH:MM format (e.g., "2:20" for 2 hours 20 minutes)')
    parser.add_argument('--end', help='End time in HH:MM format (e.g., "2:22" for 2 hours 22 minutes)')

    args = parser.parse_args()

    # Determine the URL to process
    if args.youtube:
        url = args.youtube
    elif args.x:
        url = args.x
    else:
        url = args.url

    if not url:
        parser.error("Please provide a video URL")

    # Set up logging
    logger = setup_logging('single_video')
    logger.info(f"Processing video: {url}")

    # Extract platform and video info
    platform, video_id, full_url = parse_video_url(url)

    if not platform:
        print(f"Error: Could not determine platform from URL: {url}")
        sys.exit(1)

    print("="*60, flush=True)
    print(f"Processing {platform.upper()} Video", flush=True)
    print("="*60, flush=True)
    print(f"Video ID: {video_id}", flush=True)
    print(f"Full URL: {full_url}", flush=True)

    # Set up directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if platform == 'youtube':
        download_dir = Path(f"data/youtube_downloads/{timestamp}")
        output_dir = Path("data/youtube_analysis")
    elif platform == 'direct':
        download_dir = Path(f"data/direct_downloads/{timestamp}")
        output_dir = Path("data/direct_analysis")
    else:
        download_dir = Path(f"data/x_downloads/{timestamp}")
        output_dir = Path("data/x_analysis")

    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    # Initialize database
    db = get_database()

    # Check if already processed
    db_video_id = f"{platform}_{video_id}"
    db_entry = db.get_video(db_video_id)

    if db_entry:
        print(f"\n✓ Video already in database: {db_entry.get('title', video_id)}")
        print(f"  Processed on: {db_entry.get('processed_date', 'Unknown')}")

        if not args.force:
            response = input("\nProcess anyway? (y/N): ")
            if response.lower() != 'y':
                print("Skipping processing.")
                return
        else:
            print("  Force flag set - reprocessing...")

    try:
        # Step 1: Download audio
        print("\n" + "="*60, flush=True)
        print("STEP 1: Downloading audio", flush=True)
        print("="*60, flush=True)

        if platform == 'youtube':
            if args.start and args.end:
                audio_file = download_youtube_audio(video_id, download_dir, args.start, args.end)
                # Skip DO trimming since we already have the segment
                skip_trimming = True
            else:
                audio_file = download_youtube_audio(video_id, download_dir)
                skip_trimming = False
        elif platform == 'direct':
            audio_file = download_direct_url(full_url, download_dir)
            skip_trimming = False
        else:
            audio_file = download_x_audio(video_id, download_dir, full_url)
            skip_trimming = False

        print(f"✓ Downloaded: {audio_file.name}")

        # Step 1.5: Trim audio if time range specified (skip for YouTube if already segmented)
        if (args.start or args.end) and not skip_trimming:
            if not args.start:
                args.start = "0:00:00"
            if not args.end:
                # If no end specified, we'll need to get duration (or use a large value)
                args.end = "99:99:99"  # Will be clamped to actual duration by ffmpeg

            print("\n" + "="*60)
            print("STEP 1.5: Trimming audio")
            print("="*60)

            audio_file = trim_audio_on_do(audio_file, args.start, args.end, download_dir)
            print(f"✓ Using trimmed audio: {audio_file.name}")

        # Step 2: Transcribe
        print("\n" + "="*60)
        print("STEP 2: Transcribing audio")
        print("="*60)

        # Use sanitized filename to match what transcription_runner produces
        sanitized_transcript_name = sanitize_filename(f"{audio_file.stem}_transcript.json")
        transcript_file = transcripts_dir / sanitized_transcript_name

        if transcript_file.exists():
            print(f"Transcript already exists: {transcript_file}")
        else:
            if args.local_transcribe:
                # Use local Whisper Docker container
                print("Using local Whisper Docker container...")
                runner = LocalWhisperRunner(model="base")

                # Setup Docker if needed
                if not runner.setup():
                    logger.error("Failed to setup Docker environment")
                    sys.exit(1)

                # Transcribe the audio
                result = runner.transcribe_file(
                    str(audio_file),
                    output_path=str(transcript_file),
                    output_format="json"
                )

                print(f"✓ Transcript saved to: {transcript_file}")
            else:
                # Use Vast.ai
                vast_api_key = get_vast_api_key()

                if not vast_api_key:
                    print("Error: No Vast.ai API key found. Use --local-transcribe for local processing.")
                    sys.exit(1)

                transcript_file = Path(transcribe_audio(audio_file, transcripts_dir, vast_api_key))

        # Step 3: Post-process transcript
        print("\n" + "="*60)
        print("STEP 3: Post-processing transcript")
        print("="*60)

        # Use sanitized filename for postprocessed file too
        sanitized_postprocessed_name = sanitize_filename(f"{audio_file.stem}_transcript_claude_postprocessed.json")
        postprocessed_file = transcripts_dir / sanitized_postprocessed_name

        if postprocessed_file.exists():
            print(f"Post-processed transcript already exists: {postprocessed_file}")
        else:
            try:
                postprocessed_file = postprocess_transcript_claude(
                    transcript_path=transcript_file
                )
                print("✓ Transcript post-processing completed with Claude 3 Haiku")
            except Exception as e:
                print(f"⚠ Post-processing failed: {e}")
                print("  Continuing with original transcript...")
                postprocessed_file = None

        # Use postprocessed transcript if available
        final_transcript = postprocessed_file if postprocessed_file and postprocessed_file.exists() else transcript_file

        # Step 4: Log to database
        print("\n" + "="*60)
        print("STEP 4: Updating database")
        print("="*60)

        text = extract_text_from_transcript(final_transcript)

        # Log the video
        db_result = log_video(
            transcript_path=final_transcript,
            title=audio_file.stem.replace('_', ' '),
            url=full_url,
            platform=platform,
            category='general',
            source_script='process_single_video.py',
            processing_stats={
                'word_count': len(text.split()),
                'char_count': len(text),
                'vast_ai_cost': 0.10 if not args.local_transcribe else 0,
            },
            metadata={
                'video_id': video_id,
                'processed_date': datetime.now().isoformat()
            }
        )

        # Log post-processing if it happened
        if postprocessed_file and postprocessed_file.exists():
            log_postprocessing(
                video_id=db_result,
                postprocessed_path=postprocessed_file,
                model='claude-3-haiku',
                cost=0.02
            )

        print(f"✓ Added to database: {db_result}")

        # Summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Platform: {platform.upper()}")
        print(f"Video ID: {video_id}")
        print(f"Audio: {audio_file}")
        print(f"Transcript: {final_transcript}")
        print(f"Database ID: {db_result}")

        logger.info(f"Processing complete for {platform} video {video_id}")

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
