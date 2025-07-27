#!/usr/bin/env python3
"""
Step 1: Download and transcribe Rogue Trader episodes
This script ONLY downloads and transcribes - no analysis yet
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from downloaders.youtube_channel_downloader import YouTubeChannelDownloader
from infrastructure.vast_ai.config import WHISPER_MODEL


def main():
    # Configuration
    channel_url = "https://www.youtube.com/@roguetrader100"
    local_download_dir = Path("data/episodes/rogue_trader_downloads")
    output_dir = Path("data/episodes/rogue_trader_analysis")
    
    local_download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download 10 episodes
    print("="*80)
    print("STEP 1: Downloading 10 episodes from Rogue Trader")
    print("="*80)
    
    downloader = YouTubeChannelDownloader(str(local_download_dir))
    
    print(f"Downloading from {channel_url}...")
    download_result = downloader.download_channel(channel_url, max_videos=10)
    
    if 'new_downloads' in download_result:
        print(f"Downloaded {download_result['new_downloads']} new episodes")
    
    audio_files = [Path(f) for f in downloader.get_downloaded_files() if Path(f).exists()]
    print(f"Found {len(audio_files)} episodes")
    
    if not audio_files:
        print("No episodes found to process")
        return
    
    # Limit to 10 most recent
    audio_files = audio_files[:10]
    print(f"\nWill transcribe {len(audio_files)} episodes:")
    for i, f in enumerate(audio_files):
        print(f"  {i+1}. {f.name}")
    
    # Get Vast.ai API key
    vast_api_key = os.getenv("VAST_API_KEY")
    if not vast_api_key:
        api_key_file = Path.home() / ".config/vastai/vast_api_key"
        if api_key_file.exists():
            vast_api_key = api_key_file.read_text().strip()
    
    if not vast_api_key:
        print("\nError: No Vast.ai API key found")
        print("Please set VAST_API_KEY environment variable or create ~/.config/vastai/vast_api_key")
        return
    
    # Step 2: Transcribe each episode
    print("\n" + "="*80)
    print("STEP 2: Transcribing episodes")
    print("="*80)
    
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    
    transcribed_files = []
    
    # Check which episodes need transcription
    episodes_to_transcribe = []
    for audio_file in audio_files:
        transcript_file = transcripts_dir / f"{audio_file.stem}_transcript.json"
        if transcript_file.exists():
            print(f"✓ {audio_file.name} - transcript exists")
            transcribed_files.append(transcript_file)
        else:
            episodes_to_transcribe.append(audio_file)
    
    if not episodes_to_transcribe:
        print("\nAll episodes already transcribed!")
    else:
        print(f"\nNeed to transcribe {len(episodes_to_transcribe)} episodes")
        
        from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
        
        vast_runner = TranscriptionRunner(vast_api_key)
        
        try:
            # Set up GPU instance ONCE
            print(f"\nStarting Vast.ai GPU instance (using Whisper model: {WHISPER_MODEL})...")
            instance = vast_runner.setup_instance(
                gpu_type="RTX 3080",
                max_price=0.30
            )
            print(f"GPU instance {instance['id']} ready")
            
            # Transcribe all episodes with the same instance
            for i, audio_file in enumerate(episodes_to_transcribe):
                print(f"\nTranscribing {i+1}/{len(episodes_to_transcribe)}: {audio_file.name}")
                
                transcript_file = transcripts_dir / f"{audio_file.stem}_transcript.json"
                
                try:
                    # Transcribe
                    transcript_result = vast_runner.transcribe_audio(
                        audio_path=audio_file,
                        output_dir=transcripts_dir,
                        model=WHISPER_MODEL,
                        use_faster_whisper=True
                    )
                    
                    print(f"✓ Completed in {transcript_result['metadata']['transcription_time']:.1f}s")
                    transcribed_files.append(transcript_file)
                    
                except Exception as e:
                    print(f"✗ Error transcribing {audio_file.name}: {e}")
            
        finally:
            # Cleanup GPU only after ALL transcriptions
            print("\nCleaning up GPU instance...")
            vast_runner.cleanup(destroy_instance=True)
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD AND TRANSCRIPTION COMPLETE")
    print("="*80)
    print(f"Downloaded episodes: {len(audio_files)}")
    print(f"Transcribed episodes: {len(transcribed_files)}")
    print(f"Transcripts saved to: {transcripts_dir}")
    
    # Save metadata for next step
    metadata_file = output_dir / "transcription_metadata.json"
    metadata = {
        'channel': 'Rogue Trader',
        'channel_url': channel_url,
        'episodes_downloaded': len(audio_files),
        'episodes_transcribed': len(transcribed_files),
        'transcript_files': [str(f) for f in transcribed_files],
        'transcription_date': datetime.now().isoformat()
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    print("\nNext step: Run analyze_rogue_trader.py to analyze these transcripts")


if __name__ == "__main__":
    main()