"""
Configuration module for video transcription system
"""
import os
from pathlib import Path

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
OPENAI_API_BASE = "https://api.openai.com/v1"
DIGITAL_OCEAN_API_KEY = os.getenv("DIGITAL_OCEAN_API_KEY", "dop_v1_a9a440c82d4d6eb634e493f464db83b487a2c4949748f608b01e832255c19268")

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

# LLM Configuration for Prediction Extraction
# Available models and their context windows:
# - "gpt-4-turbo" (128k tokens) - Best accuracy, most expensive
# - "gpt-4o" (128k tokens) - Good balance of speed/accuracy/cost
# - "gpt-4o-mini" (128k tokens) - Fastest and cheapest, lower accuracy
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")  # Default to GPT-4o for better cost/performance balance

LLM_TEMPERATURE = 0  # Zero temperature for maximum consistency in extraction
LLM_MAX_TOKENS = 2000

# Model-specific chunk sizes to avoid rate limits
# Your account has 30k TPM limit, so we need smaller chunks
# 28k tokens = ~112k chars, leaving 2k tokens for prompt/response
MODEL_CHUNK_SIZES = {
    "gpt-4-turbo": 112000,  # ~28k tokens, leaving room for prompt
    "gpt-4o": 112000,       # ~28k tokens, leaving room for prompt  
    "gpt-4o-mini": 112000,  # ~28k tokens, leaving room for prompt
}

# Get chunk size based on current model
CHUNK_SIZE = MODEL_CHUNK_SIZES.get(LLM_MODEL, 120000)
CHUNK_OVERLAP = 20000  # Increased overlap to avoid missing predictions at boundaries

# Ensure directories exist
DOWNLOADS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
