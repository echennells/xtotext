"""
Configuration for Local Whisper transcription
"""
import os
from pathlib import Path

# Docker configuration
DOCKER_IMAGE_NAME = os.getenv("LOCAL_WHISPER_IMAGE", "whisper-cpp-apple-silicon")
DOCKERFILE_PATH = os.getenv("LOCAL_WHISPER_DOCKERFILE", "metal/Dockerfile")

# Model configuration
DEFAULT_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")
DEFAULT_LANGUAGE = os.getenv("LOCAL_WHISPER_LANGUAGE", "en")

# Available models (in order of size/quality)
AVAILABLE_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3"
]

# Performance settings
DEFAULT_THREADS = os.getenv("LOCAL_WHISPER_THREADS", "4")
DEFAULT_PROCESSORS = os.getenv("LOCAL_WHISPER_PROCESSORS", "1")

# Output settings
DEFAULT_OUTPUT_FORMAT = os.getenv("LOCAL_WHISPER_OUTPUT_FORMAT", "json")
SUPPORTED_OUTPUT_FORMATS = ["json", "txt", "vtt", "srt", "csv"]

# Additional whisper-cli arguments
ADDITIONAL_ARGS = os.getenv("LOCAL_WHISPER_ADDITIONAL_ARGS", "").split() if os.getenv("LOCAL_WHISPER_ADDITIONAL_ARGS") else []