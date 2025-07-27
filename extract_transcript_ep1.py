#!/usr/bin/env python3
"""
Quick script to extract transcript from Episode 1
"""
import json
import sys
from pathlib import Path

# Add imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from infrastructure.vast_ai import InstanceManager, TranscriptionRunner
from infrastructure.vast_ai.config import VAST_CONFIG

# Episode 1 file
audio_file = Path("data/episodes/bitcoin_dive_bar_analysis/Bitcoin Dive Bar EP 01 - Bitcoin All Time Highs_iuCuCG-4V7E.opus")

if not audio_file.exists():
    print(f"Audio file not found: {audio_file}")
    sys.exit(1)

print("Setting up GPU instance...")
instance_manager = InstanceManager()
runner = TranscriptionRunner(instance_manager)

# Find or create instance
instance = instance_manager.find_or_create_instance(
    gpu_type="RTX 3080",
    max_price=0.30
)
print(f"Using instance {instance['id']}")

# Transcribe
print(f"Transcribing {audio_file.name}...")
result = runner.transcribe_file(str(audio_file))

if result['status'] == 'success':
    # Save transcript locally
    transcript_file = audio_file.parent / "transcripts" / f"{audio_file.stem}_transcript.json"
    transcript_file.parent.mkdir(exist_ok=True)
    
    with open(transcript_file, 'w') as f:
        json.dump(result['transcript'], f, indent=2)
    
    print(f"Transcript saved to: {transcript_file}")
    
    # Find the segment around timestamp 4753
    segments = result['transcript'].get('segments', [])
    target = 4753
    
    print(f"\nContext around timestamp {target}s (1:19:13):")
    print("-" * 80)
    
    for seg in segments:
        if seg['start'] <= target + 30 and seg['end'] >= target - 30:
            print(f"[{seg['start']:.1f}-{seg['end']:.1f}s]: {seg['text']}")
else:
    print(f"Transcription failed: {result.get('error')}")

# Cleanup
instance_manager.stop_instance(instance['id'])