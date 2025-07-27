#!/usr/bin/env python3
"""
NeMo Speaker Diarization and Separation Script
Separates speakers from audio file into individual files
"""

import os
import json
import torch
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
import soundfile as sf
import librosa

def create_manifest(audio_file_path, output_dir):
    """Create manifest file for NeMo diarization"""
    manifest_path = os.path.join(output_dir, "input_manifest.json")
    
    # Get audio info
    audio_info = sf.info(audio_file_path)
    duration = audio_info.duration
    
    # Create manifest entry
    manifest_entry = {
        "audio_filepath": audio_file_path,
        "offset": 0,
        "duration": duration,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None
    }
    
    # Write manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest_entry, f)
        f.write('\n')
    
    print(f"Created manifest: {manifest_path}")
    return manifest_path

def create_nemo_config(output_dir):
    """Create NeMo diarization config"""
    
    config = {
        "diarizer": {
            "manifest_filepath": None,  # We'll use paths2audio_files instead
            "out_dir": output_dir,
            "oracle_vad": False,  # We don't have ground truth VAD
            "collar": 0.25,
            "ignore_overlap": True,
            
            # VAD settings
            "vad": {
                "model_path": "vad_multilingual_marblenet",  # Pretrained VAD model
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.875,
                    "onset": 0.8,
                    "offset": 0.6,
                    "pad_onset": 0.05,
                    "pad_offset": -0.05,
                    "min_duration_on": 0.2,
                    "min_duration_off": 0.2
                }
            },
            
            # Speaker embedding settings
            "speaker_embeddings": {
                "model_path": "titanet_large",  # Pretrained speaker model
                "parameters": {
                    "window_length_in_sec": 1.5,
                    "shift_length_in_sec": 0.75,
                    "multiscale_weights": [1, 1, 1, 1, 1],
                    "save_embeddings": True
                }
            },
            
            # Clustering settings
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": 8,  # Reasonable for a podcast
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30
                }
            }
        },
        
        # System settings
        "num_workers": 0,  # Fix for multiprocessing/tensor issues
        "sample_rate": 16000,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    return OmegaConf.create(config)

def extract_speaker_segments(audio_file, rttm_file, output_dir):
    """Extract individual speaker segments based on RTTM file"""
    
    # Load audio
    print(f"Loading audio: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Parse RTTM file
    speaker_segments = {}
    
    with open(rttm_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 8:
                    # RTTM format: SPEAKER file 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]
                    
                    if speaker_id not in speaker_segments:
                        speaker_segments[speaker_id] = []
                    
                    speaker_segments[speaker_id].append({
                        'start': start_time,
                        'duration': duration,
                        'end': start_time + duration
                    })
    
    print(f"Found {len(speaker_segments)} speakers: {list(speaker_segments.keys())}")
    
    # Extract and save each speaker's audio
    speaker_files = {}
    for speaker_id, segments in speaker_segments.items():
        print(f"Processing speaker {speaker_id} with {len(segments)} segments...")
        
        # Concatenate all segments for this speaker
        speaker_audio = []
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = audio[start_sample:end_sample]
            speaker_audio.append(segment_audio)
        
        if speaker_audio:
            # Concatenate all segments
            full_speaker_audio = np.concatenate(speaker_audio)
            
            # Save speaker file
            speaker_filename = f"speaker_{speaker_id}.wav"
            speaker_path = os.path.join(output_dir, speaker_filename)
            sf.write(speaker_path, full_speaker_audio, sr)
            
            duration_mins = len(full_speaker_audio) / sr / 60
            print(f"Saved {speaker_filename}: {duration_mins:.1f} minutes")
            speaker_files[speaker_id] = speaker_path
    
    return speaker_files

def main():
    # Configuration
    input_audio = "Bitcoin Dive Bar EP 01 - Bitcoin All Time Highs_iuCuCG-4V7E.opus"
    output_dir = "speaker_separation_output"
    
    # Check if audio file exists
    if not os.path.exists(input_audio):
        print(f"ERROR: Audio file not found: {input_audio}")
        print("Please make sure the file is in the current directory.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Convert OPUS to WAV first (NeMo has issues with OPUS)
    print("\n=== Step 0: Converting OPUS to WAV ===")
    wav_audio = os.path.join(output_dir, "input_audio.wav")
    if not os.path.exists(wav_audio):
        print(f"Converting {input_audio} to WAV format...")
        audio, sr = librosa.load(input_audio, sr=16000)  # NeMo expects 16kHz
        sf.write(wav_audio, audio, sr)
        print(f"Converted audio saved as: {wav_audio}")
    else:
        print(f"Using existing converted audio: {wav_audio}")
    
    # Use the converted WAV file from here on
    input_audio = wav_audio
    
    try:
        # Step 1: Create config (skip manifest since using direct paths)
        print("\n=== Step 1: Creating config ===")
        config = create_nemo_config(output_dir)
        
        # Step 2: Initialize diarizer
        print("\n=== Step 2: Initializing NeMo ClusteringDiarizer ===")
        print("This will download pretrained models (VAD and speaker embedding)...")
        
        diarizer = ClusteringDiarizer(cfg=config)
        
        # Step 3: Run diarization
        print("\n=== Step 3: Running speaker diarization ===")
        print("This may take several minutes for an 85-minute podcast...")
        
        # Use paths2audio_files instead of manifest - simpler and more reliable
        print("Using direct file path method...")
        diarizer.diarize(paths2audio_files=[input_audio])
        
        # Step 4: Find the generated RTTM file
        print("\n=== Step 4: Processing results ===")
        rttm_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.rttm'):
                    rttm_files.append(os.path.join(root, file))
        
        if not rttm_files:
            print("ERROR: No RTTM file generated!")
            return
        
        rttm_file = rttm_files[0]  # Use the first RTTM file found
        print(f"Using RTTM file: {rttm_file}")
        
        # Step 5: Extract speaker segments
        print("\n=== Step 5: Extracting individual speaker files ===")
        import numpy as np  # Import here since we need it for concatenation
        
        speaker_files = extract_speaker_segments(input_audio, rttm_file, output_dir)
        
        # Summary
        print("\n=== COMPLETE! ===")
        print(f"Generated {len(speaker_files)} speaker files:")
        for speaker_id, file_path in speaker_files.items():
            print(f"  Speaker {speaker_id}: {file_path}")
        
        print(f"\nAll files saved in: {output_dir}")
        print("\nNext steps:")
        print("1. Listen to each speaker file")
        print("2. Rename them to actual names (e.g., speaker_0.wav -> john_host.wav)")
        print("3. Use these labeled files for Stage 2 (speaker identification training)")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
