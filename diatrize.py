import os
import json
import librosa
import soundfile as sf
from omegaconf import OmegaConf

# Use our good 10-minute chunk
chunk_file = "temp_chunks/chunk_debug.wav"
audio, sr = librosa.load(chunk_file, sr=16000)
print(f"Processing chunk: {len(audio)/sr/60:.1f} minutes")

# Create manifest
abs_chunk_path = os.path.abspath(chunk_file)
manifest = {
    "audio_filepath": abs_chunk_path,
    "duration": len(audio) / sr,
    "label": "infer"
}

manifest_file = "temp_chunks/input_manifest.json"
with open(manifest_file, 'w') as f:
    json.dump(manifest, f)
    f.write('\n')

print("Starting basic clustering diarization...")
from nemo.collections.asr.models import ClusteringDiarizer

# Fix: smoothing should be a string method name
cfg = {
    'device': 'cuda',
    'num_workers': 4,
    'sample_rate': 16000,
    'verbose': True,
    'diarizer': {
        'manifest_filepath': os.path.abspath(manifest_file),
        'out_dir': os.path.abspath('temp_chunks/basic_outputs'),
        'oracle_vad': False,
        
        'vad': {
            'model_path': 'vad_telephony_marblenet',
            'parameters': {
                'window_length_in_sec': 0.15,
                'shift_length_in_sec': 0.01,
                'smoothing': 'median',  # Change to string: 'median' or 'mean'
                'overlap': 0.5,
                'onset': 0.8,
                'offset': 0.6,
                'pad_onset': 0.05,
                'pad_offset': -0.1,
                'min_duration_on': 0.2,
                'min_duration_off': 0.2,
                'overlap_ratio': 0.875,
            }
        },
        
        'speaker_embeddings': {
            'model_path': 'nvidia/speakerverification_en_titanet_large',
            'parameters': {
                'window_length_in_sec': 1.5,
                'shift_length_in_sec': 0.75,
                'multiscale_weights': [1, 1, 1, 1, 1],
                'save_embeddings': False,
            }
        },
        
        'clustering': {
            'parameters': {
                'oracle_num_speakers': None,
                'max_num_speakers': 8,
                'enhanced_count_thres': 80,
            }
        }
    }
}

os.makedirs("temp_chunks/basic_outputs", exist_ok=True)

# Convert to OmegaConf
config = OmegaConf.create(cfg)

print("Initializing ClusteringDiarizer...")
diarizer = ClusteringDiarizer(cfg=config)

print("Running diarization...")
diarizer.diarize()

print("Diarization complete! Checking outputs...")

# Look for results
output_dir = "temp_chunks/basic_outputs"
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        print(f"Output file: {file}")
        if file.endswith('.rttm'):
            rttm_path = os.path.join(output_dir, file)
            with open(rttm_path, 'r') as f:
                lines = f.readlines()[:10]  # First 10 lines
            print("First few speaker segments:")
            for line in lines:
                print(line.strip())
