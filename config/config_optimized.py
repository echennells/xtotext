"""
Optimized configuration for reduced memory/IO footprint
"""
import os
from pathlib import Path

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = "https://api.openai.com/v1"

# Directory Configuration
BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
TEMP_DIR = BASE_DIR / "temp"

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Transcription Configuration
WHISPER_MODEL = "whisper-1"
TRANSCRIPTION_LANGUAGE = "en"
COST_PER_MINUTE = 0.006  # $0.006 per minute for Whisper API

# Video Download Configuration
VIDEO_FORMAT = "best[ext=mp4]/best"
DEFAULT_OUTPUT_PATTERN = "%(title)s.%(ext)s"

# Optimized LLM Configuration for Prediction Extraction
LLM_MODEL = "gpt-4o-mini"  # OpenAI GPT-4o-mini
LLM_TEMPERATURE = 0.1  # Low temperature for consistent extraction
LLM_MAX_TOKENS = 2000

# Optimized chunking settings to reduce memory pressure and API calls
CHUNK_SIZE = 8000  # Larger chunks = fewer API calls, less memory overhead
CHUNK_OVERLAP = 500  # Reduced overlap from 1000 to 500 (6.25% vs 33% redundancy)

# Memory optimization settings
MAX_MEMORY_MB = 1024  # Maximum memory usage before triggering garbage collection
ENABLE_STREAMING = True  # Enable streaming JSON parsing for large files
USE_GENERATOR_CHUNKING = True  # Use generator-based chunking to reduce memory

# IO optimization settings
BATCH_WRITE_SIZE = 10  # Write predictions in batches instead of all at once
ENABLE_COMPRESSION = True  # Compress large transcript files after processing

# Processing optimization
PARALLEL_CHUNKS = False  # Process chunks sequentially to reduce memory spikes
CHUNK_PROCESSING_DELAY_MS = 100  # Small delay between chunks to prevent IO spikes

# Ensure directories exist
DOWNLOADS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)