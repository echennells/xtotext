#!/usr/bin/env python3
"""
One-off script to download and transcribe a single YouTube video
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from downloaders.youtube_channel_downloader import YouTubeChannelDownloader
from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
from infrastructure.vast_ai.config import WHISPER_MODEL


def main():
    video_url = "https://www.youtube.com/watch?v=UTsBQ-aVx1s"
    
    # Create directories
    download_dir = Path("data/one_off_downloads")
    transcript_dir = Path("data/one_off_transcripts")
    download_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download the video
    print("="*80)
    print("STEP 1: Downloading YouTube video")
    print("="*80)
    print(f"URL: {video_url}")
    
    # Use yt-dlp directly for single video
    import subprocess
    
    # Extract video info first
    cmd = ["yt-dlp", "--print", "%(title)s", video_url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_title = result.stdout.strip()
        print(f"Video title: {video_title}")
    except subprocess.CalledProcessError as e:
        print(f"Error getting video info: {e}")
        return
    
    # Download audio
    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "opus",
        "--audio-quality", "0",
        "-o", str(download_dir / "%(title)s_%(id)s.%(ext)s"),
        video_url
    ]
    
    try:
        print("\nDownloading audio...")
        subprocess.run(cmd, check=True)
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return
    
    # Find the downloaded file
    audio_files = list(download_dir.glob("*.opus")) + list(download_dir.glob("*.m4a")) + list(download_dir.glob("*.mp3"))
    
    if not audio_files:
        print("No audio file found after download")
        return
    
    audio_file = audio_files[0]
    print(f"Downloaded: {audio_file.name}")
    
    # Check if transcript already exists
    transcript_file = transcript_dir / f"{audio_file.stem}_transcript.json"
    
    if transcript_file.exists():
        print(f"\nTranscript already exists: {transcript_file}")
        print("Delete it if you want to re-transcribe.")
        return
    
    # Get Vast.ai API key
    vast_api_key = os.getenv("VAST_API_KEY")
    if not vast_api_key:
        api_key_file = Path.home() / ".config/vastai/vast_api_key"
        if api_key_file.exists():
            vast_api_key = api_key_file.read_text().strip()
    
    if not vast_api_key:
        print("Error: No Vast.ai API key found (needed for GPU transcription)")
        return
    
    # Step 2: Transcribe with Vast.ai
    print("\n" + "="*80)
    print("STEP 2: Transcribing with Vast.ai GPU")
    print("="*80)
    
    vast_runner = TranscriptionRunner(vast_api_key)
    
    try:
        # Set up GPU instance
        print("Starting Vast.ai GPU instance...")
        instance = vast_runner.setup_instance(
            gpu_type="RTX 3080",
            max_price=0.30
        )
        print(f"GPU instance {instance['id']} ready")
        
        # Transcribe
        transcript_result = vast_runner.transcribe_audio(
            audio_path=audio_file,
            output_dir=transcript_dir,
            model=WHISPER_MODEL,
            use_faster_whisper=True
        )
        
        print(f"\nTranscription completed in {transcript_result['metadata']['transcription_time']:.1f}s")
        print(f"Transcript saved to: {transcript_file}")
        
    finally:
        # Cleanup GPU immediately after use
        print("\nCleaning up Vast.ai GPU instance...")
        vast_runner.cleanup(destroy_instance=True)
    
    print("\n" + "="*80)
    print("DONE! Transcript is ready.")
    print("="*80)


if __name__ == "__main__":
    main()