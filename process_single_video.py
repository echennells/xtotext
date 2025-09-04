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
import re

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
from infrastructure.digital_ocean.simple_runner import SimpleDigitalOceanRunner
from processors.claude_transcript_postprocessor import postprocess_transcript_claude
from database import get_database, log_video, log_postprocessing
from downloaders.x_downloader import XDownloader
from downloaders.youtube_channel_downloader import YouTubeChannelDownloader
from utils.filename_utils import sanitize_filename
import subprocess


def setup_logging():
    """Set up logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"single_video_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def extract_video_info(url):
    """Extract platform and video ID from URL"""
    # YouTube patterns
    youtube_patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'  # Just the video ID
    ]
    
    # X/Twitter patterns
    x_patterns = [
        r'(?:twitter\.com|x\.com)/\w+/status/(\d+)',
        r'^(\d{15,20})$'  # Just the tweet ID
    ]
    
    # Check YouTube
    for pattern in youtube_patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            return 'youtube', video_id, f"https://www.youtube.com/watch?v={video_id}"
    
    # Check X/Twitter
    for pattern in x_patterns:
        match = re.search(pattern, url)
        if match:
            tweet_id = match.group(1)
            return 'x', tweet_id, f"https://x.com/i/status/{tweet_id}"
    
    return None, None, None


def download_youtube(video_id, download_dir):
    """Download YouTube video audio using YouTubeChannelDownloader"""
    print(f"Downloading YouTube video: {video_id}")
    
    # Check if we already have the audio file
    existing_files = list(Path("data").rglob(f"*{video_id}*.mp3"))
    if existing_files:
        print(f"✓ Found existing audio file: {existing_files[0]}")
        return existing_files[0]
    
    # Use YouTubeChannelDownloader for consistency with main.py
    downloader = YouTubeChannelDownloader(str(download_dir))
    
    # Use the new public method
    success, filepath = downloader.download_single_video_by_id(video_id)
    
    if not success or not filepath:
        raise RuntimeError(f"Failed to download video {video_id}")
    
    return Path(filepath)


def trim_audio_on_do(audio_file, start_time, end_time, output_dir):
    """
    Trim audio file using Digital Ocean droplet to offload processing
    
    Args:
        audio_file: Path to the audio file
        start_time: Start time string (e.g., "2:20:00")
        end_time: End time string (e.g., "2:22:00")
        output_dir: Directory for trimmed output
    
    Returns:
        Path to trimmed audio file
    """
    print(f"\n{'='*60}")
    print("TRIMMING AUDIO ON DIGITAL OCEAN")
    print(f"{'='*60}")
    print(f"Start: {start_time}, End: {end_time}")
    
    # Initialize DO runner
    do_runner = SimpleDigitalOceanRunner()
    
    # Upload audio file to DO
    remote_audio = f"/workspace/audio/{audio_file.name}"
    print(f"Uploading {audio_file.name} to Digital Ocean...")
    do_runner.upload_file(str(audio_file), remote_audio)
    
    # Prepare output filename
    trimmed_name = f"trimmed_{audio_file.stem}_{start_time.replace(':', '')}_{end_time.replace(':', '')}{audio_file.suffix}"
    remote_trimmed = f"/workspace/audio/{trimmed_name}"
    
    # Run ffmpeg on DO
    ffmpeg_cmd = f'ffmpeg -i "{remote_audio}" -ss {start_time} -to {end_time} -c copy "{remote_trimmed}" -y'
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


def download_x(tweet_id, download_dir):
    """Download X/Twitter video audio"""
    print(f"Downloading X/Twitter video: {tweet_id}")
    
    # Check if we already have the audio file (including in timestamped subdirs)
    existing_files = list(Path("data").rglob(f"*{tweet_id}*.mp3")) + \
                    list(Path("data").rglob(f"*{tweet_id}*.m4a"))
    
    if existing_files:
        # Use the first existing file
        existing_file = existing_files[0]
        print(f"Found existing audio file: {existing_file}")
        
        # Copy to the timestamp directory for consistency
        import shutil
        download_dir.mkdir(parents=True, exist_ok=True)
        target_file = download_dir / existing_file.name
        shutil.copy2(existing_file, target_file)
        print(f"Copied to: {target_file}")
        return target_file
    
    # If not found, try downloading
    downloader = XDownloader(str(download_dir), download_timeout=1800)  # 30 minute timeout
    
    # Construct the X URL
    x_url = f"https://x.com/i/status/{tweet_id}"
    
    # Download the audio
    audio_file = downloader.download_audio(x_url)
    
    if not audio_file:
        raise RuntimeError(f"Failed to download audio from tweet {tweet_id}")
    
    if not audio_file.exists():
        raise RuntimeError(f"Downloaded audio file not found: {audio_file}")
    
    return audio_file


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
    print("DEBUG: Entering main()", flush=True)
    parser = argparse.ArgumentParser(description='Process a single video through the full pipeline')
    parser.add_argument('url', nargs='?', help='Video URL (YouTube or X/Twitter)')
    parser.add_argument('--youtube', help='YouTube video URL or ID')
    parser.add_argument('--x', '--twitter', help='X/Twitter status URL or ID')
    parser.add_argument('--local-transcribe', action='store_true', help='Use local Whisper instead of Vast.ai')
    parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation prompts and force reprocessing')
    parser.add_argument('--start', help='Start time for trimming (e.g., "2:20:00" or "140:00" or "8400")')
    parser.add_argument('--end', help='End time for trimming (e.g., "2:22:00" or "142:00" or "8520")')
    
    print("DEBUG: Parsing arguments...", flush=True)
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: {args}", flush=True)
    
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
    logger = setup_logging()
    logger.info(f"Processing video: {url}")
    print(f"DEBUG: Starting processing for {url}", flush=True)
    
    # Extract platform and video info
    print("DEBUG: Extracting video info...", flush=True)
    platform, video_id, full_url = extract_video_info(url)
    print(f"DEBUG: Platform={platform}, ID={video_id}", flush=True)
    
    if not platform:
        print(f"Error: Could not determine platform from URL: {url}")
        sys.exit(1)
    
    print("="*60)
    print(f"Processing {platform.upper()} Video")
    print("="*60)
    print(f"Video ID: {video_id}")
    print(f"Full URL: {full_url}")
    
    # Set up directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if platform == 'youtube':
        download_dir = Path(f"data/youtube_downloads/{timestamp}")
        output_dir = Path("data/youtube_analysis")
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
        print("\n" + "="*60)
        print("STEP 1: Downloading audio")
        print("="*60)
        
        if platform == 'youtube':
            audio_file = download_youtube(video_id, download_dir)
        else:
            audio_file = download_x(video_id, download_dir)
        
        print(f"✓ Downloaded: {audio_file.name}")
        
        # Step 1.5: Trim audio if time range specified
        if args.start or args.end:
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
                # Use local Whisper
                print("Using local Whisper model...")
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(str(audio_file))
                
                with open(transcript_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"✓ Transcript saved to: {transcript_file}")
            else:
                # Use Vast.ai
                vast_api_key = os.getenv("VAST_API_KEY")
                if not vast_api_key:
                    try:
                        from config.config import VAST_API_KEY
                        vast_api_key = VAST_API_KEY
                    except ImportError:
                        pass
                
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
        
        # Calculate stats
        with open(final_transcript, 'r') as f:
            transcript_data = json.load(f)
        text = transcript_data.get('text', '')
        if not text and 'segments' in transcript_data:
            text = ' '.join(seg.get('text', '') for seg in transcript_data['segments'])
        
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