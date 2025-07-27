#!/usr/bin/env python3
"""
Rogue Trader Analysis - Analyze finance/trading content to generate better prompts
for Bitcoin Dive Bar prediction extraction

Flow:
1. Download 10 episodes from Rogue Trader channel
2. Transcribe them using Vast.ai
3. Analyze transcripts with stage1 model
4. Generate improved prompts for stage1 and stage2
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from infrastructure.digital_ocean.simple_runner import SimpleDigitalOceanRunner
from downloaders.youtube_channel_downloader import YouTubeChannelDownloader
from infrastructure.vast_ai.config import WHISPER_MODEL


def main():
    # Set up logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"rogue_trader_analysis_{log_timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Rogue Trader analysis - Log file: {log_file}")
    
    # Configuration
    channel_url = "https://www.youtube.com/@roguetrader100"
    local_download_dir = Path("data/episodes/rogue_trader_downloads")
    output_dir = Path("data/episodes/rogue_trader_analysis")
    
    local_download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download 10 YouTube files from Rogue Trader
    print("="*80)
    print("STEP 1: Downloading 10 episodes from Rogue Trader")
    print("="*80)
    logger.info("="*80)
    logger.info("STEP 1: Downloading 10 episodes from Rogue Trader")
    logger.info("="*80)
    
    downloader = YouTubeChannelDownloader(str(local_download_dir))
    
    print(f"Downloading from {channel_url}...")
    download_result = downloader.download_channel(channel_url, max_videos=10)
    
    if 'new_downloads' in download_result:
        print(f"Downloaded {download_result['new_downloads']} new episodes")
    
    audio_files = [Path(f) for f in downloader.get_downloaded_files() if Path(f).exists()]
    print(f"Found {len(audio_files)} episodes to analyze")
    
    if not audio_files:
        print("No episodes found to process")
        return
    
    # Limit to 10 most recent if we have more
    audio_files = audio_files[:10]
    
    # Get Vast.ai API key
    vast_api_key = os.getenv("VAST_API_KEY")
    if not vast_api_key:
        api_key_file = Path.home() / ".config/vastai/vast_api_key"
        if api_key_file.exists():
            vast_api_key = api_key_file.read_text().strip()
    
    if not vast_api_key:
        print("Error: No Vast.ai API key found")
        return
    
    print(f"\nWill analyze {len(audio_files)} episodes")
    
    # Create timestamp for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Results tracking
    all_transcripts = []
    
    # Step 2: Transcribe each episode
    for i, audio_file in enumerate(audio_files):
        print("\n" + "="*80)
        print(f"EPISODE {i+1}/{len(audio_files)}: {audio_file.name}")
        print("="*80)
        
        # Check if transcript already exists
        transcripts_dir = output_dir / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        transcript_file = transcripts_dir / f"{audio_file.stem}_transcript.json"
        
        if transcript_file.exists():
            print(f"Transcript already exists, loading it")
            with open(transcript_file, 'r') as f:
                transcript_data = json.load(f)
                all_transcripts.append({
                    'file': audio_file.name,
                    'transcript': transcript_data,
                    'path': transcript_file
                })
        else:
            print("\nTranscribing with Vast.ai GPU...")
            
            from infrastructure.vast_ai.transcription_runner import TranscriptionRunner
            
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
                    output_dir=transcripts_dir,
                    model=WHISPER_MODEL,
                    use_faster_whisper=True
                )
                
                print(f"Transcription completed in {transcript_result['metadata']['transcription_time']:.1f}s")
                
                all_transcripts.append({
                    'file': audio_file.name,
                    'transcript': transcript_result,
                    'path': transcript_file
                })
                
            finally:
                # Cleanup GPU immediately after use
                print("Cleaning up Vast.ai GPU instance...")
                vast_runner.cleanup(destroy_instance=True)
    
    # Step 3: Analyze transcripts to generate prompts
    print("\n" + "="*80)
    print("STEP 3: Analyzing Rogue Trader content to generate prompts")
    print("="*80)
    
    # Save all transcripts for analysis
    analysis_dir = output_dir / "prompt_generation"
    analysis_dir.mkdir(exist_ok=True)
    
    # Create a combined analysis file
    combined_file = analysis_dir / f"rogue_trader_transcripts_{run_timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump({
            'channel': 'Rogue Trader',
            'episodes_analyzed': len(all_transcripts),
            'analysis_date': datetime.now().isoformat(),
            'transcripts': all_transcripts
        }, f, indent=2)
    
    print(f"Saved {len(all_transcripts)} transcripts to: {combined_file}")
    
    # Now analyze with Digital Ocean
    runner = SimpleDigitalOceanRunner()
    cleanup_on_exit = True
    
    try:
        # Start DO droplet
        runner.start(droplet_size="s-2vcpu-4gb", wait_time=60)
        
        # Upload code and scripts
        print("\nUploading code to Digital Ocean...")
        runner.upload_code(Path("src"))
        runner.upload_code(Path("config"))
        
        # Upload the combined transcript file
        runner.ssh.upload_file(combined_file, "/workspace/rogue_trader_transcripts.json")
        
        # Setup environment
        print("\nSetting up Python environment...")
        setup_cmd = """
cd /workspace
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install openai tiktoken
"""
        runner.run_command(setup_cmd)
        
        # Run the prompt generator (we'll create this next)
        print("\nAnalyzing content and generating prompts...")
        analysis_cmd = """
cd /workspace
source venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY}"
python /workspace/xtotext/src/predictions/prediction_tracker/prompt_generator.py
"""
        
        result = runner.run_command(analysis_cmd, timeout=600)
        
        if result['success']:
            print("Analysis completed!")
            
            # Download generated prompts
            prompts_output = analysis_dir / f"generated_prompts_{run_timestamp}.json"
            runner.ssh.download_file(
                "/workspace/generated_prompts.json",
                prompts_output
            )
            
            print(f"\nGenerated prompts saved to: {prompts_output}")
            
            # Display the prompts
            with open(prompts_output, 'r') as f:
                prompts = json.load(f)
                print("\n" + "="*80)
                print("GENERATED PROMPTS:")
                print("="*80)
                print(json.dumps(prompts, indent=2))
                
        else:
            print(f"Analysis failed: {result['stderr']}")
            if result.get('stdout'):
                print(f"Output: {result['stdout']}")
    
    finally:
        if cleanup_on_exit and 'runner' in locals() and runner.droplet:
            print("\nCleaning up Digital Ocean droplet...")
            try:
                runner.cleanup()
                print("✓ Droplet destroyed successfully")
            except Exception as e:
                print(f"✗ Error cleaning up droplet: {e}")
    
    print("\n" + "="*80)
    print("ROGUE TRADER ANALYSIS COMPLETE")
    print("="*80)
    print(f"Transcripts: {transcripts_dir}")
    print(f"Analysis: {analysis_dir}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()