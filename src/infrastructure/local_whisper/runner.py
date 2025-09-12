"""
Simple runner for local Whisper transcription
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from .transcriber import LocalWhisperTranscriber
from .config import (
    DOCKER_IMAGE_NAME, DEFAULT_MODEL, DEFAULT_LANGUAGE,
    DEFAULT_OUTPUT_FORMAT, DOCKERFILE_PATH
)


class LocalWhisperRunner:
    """
    Simple runner for local Whisper transcription that handles:
    1. Docker image setup
    2. Audio transcription
    3. Results formatting
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        docker_image: str = DOCKER_IMAGE_NAME
    ):
        """
        Initialize the runner
        
        Args:
            model: Whisper model to use
            language: Language for transcription
            docker_image: Docker image name
        """
        self.transcriber = LocalWhisperTranscriber(
            docker_image=docker_image,
            model_name=model,
            language=language
        )
        
    def setup(self) -> bool:
        """
        Setup the Docker environment
        
        Returns:
            True if setup successful
        """
        print("Checking Docker setup...")
        
        # Check if Docker image exists
        if not self.transcriber.check_docker_image():
            print(f"Docker image {self.transcriber.docker_image} not found.")
            print("Building Docker image...")
            
            # Build the image
            if not self.transcriber.build_docker_image(DOCKERFILE_PATH):
                print("Failed to build Docker image")
                return False
        
        print("Docker setup complete!")
        return True
    
    def transcribe_file(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        output_format: str = DEFAULT_OUTPUT_FORMAT
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            output_path: Optional path to save transcript
            output_format: Format for output
            
        Returns:
            Transcript data
        """
        audio_path = Path(audio_path)
        
        print(f"Transcribing: {audio_path}")
        
        # Transcribe the file
        result = self.transcriber.transcribe(
            str(audio_path),
            output_format=output_format
        )
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_format == "json":
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    f.write(result.get("text", ""))
            
            print(f"Transcript saved to: {output_path}")
        
        return result
    
    def transcribe_directory(
        self,
        directory: str,
        output_dir: Optional[str] = None,
        extensions: List[str] = None,
        output_format: str = DEFAULT_OUTPUT_FORMAT
    ) -> Dict[str, Dict[str, Any]]:
        """
        Transcribe all audio files in a directory
        
        Args:
            directory: Directory containing audio files
            output_dir: Optional directory for saving transcripts
            extensions: List of audio file extensions to process
            output_format: Format for output
            
        Returns:
            Dictionary mapping file paths to transcript data
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Default audio extensions
        if extensions is None:
            extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.mp4']
        
        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(directory.glob(f"*{ext}"))
            audio_files.extend(directory.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            print(f"No audio files found in {directory}")
            return {}
        
        print(f"Found {len(audio_files)} audio files to transcribe")
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Transcribe all files
        results = {}
        for audio_file in audio_files:
            # Determine output path
            output_path = None
            if output_dir:
                output_name = audio_file.stem + f".{output_format}"
                output_path = output_dir / output_name
            
            # Transcribe the file
            try:
                results[str(audio_file)] = self.transcribe_file(
                    str(audio_file),
                    output_path=output_path,
                    output_format=output_format
                )
            except Exception as e:
                print(f"Failed to transcribe {audio_file}: {e}")
                results[str(audio_file)] = {"error": str(e)}
        
        return results
    
    def cleanup(self):
        """
        Cleanup resources (placeholder for future use)
        """
        pass


def run_local_transcription(
    audio_path: str,
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    output_format: str = DEFAULT_OUTPUT_FORMAT
) -> Dict[str, Any]:
    """
    Convenience function to run a single transcription
    
    Args:
        audio_path: Path to audio file
        output_path: Optional path to save transcript
        model: Whisper model to use
        language: Language for transcription
        output_format: Format for output
        
    Returns:
        Transcript data
    """
    runner = LocalWhisperRunner(model=model, language=language)
    
    # Setup Docker
    if not runner.setup():
        raise RuntimeError("Failed to setup Docker environment")
    
    # Transcribe the file
    result = runner.transcribe_file(
        audio_path,
        output_path=output_path,
        output_format=output_format
    )
    
    runner.cleanup()
    
    return result