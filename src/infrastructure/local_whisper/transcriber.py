"""
Local Whisper Transcriber using Docker container with whisper-cpp
"""
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import re


class LocalWhisperTranscriber:
    """
    Transcribes audio using local Docker container running whisper-cpp
    """
    
    def __init__(
        self,
        docker_image: str = "whisper-cpp-apple-silicon",
        model_name: str = "base",
        language: str = "en"
    ):
        """
        Initialize the local Whisper transcriber
        
        Args:
            docker_image: Name of the Docker image to use
            model_name: Whisper model size (tiny, base, small, medium, large)
            language: Language code for transcription
        """
        self.docker_image = docker_image
        self.model_name = model_name
        self.language = language
        self.model_path = f"/app/models/ggml-{model_name}.bin"
        
    def transcribe(
        self,
        audio_path: str,
        output_format: str = "json",
        additional_args: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file using the Docker container
        
        Args:
            audio_path: Path to the audio file to transcribe
            output_format: Output format (json, txt, vtt, srt)
            additional_args: Additional arguments to pass to whisper-cli
            
        Returns:
            Dictionary containing transcript and metadata
        """
        audio_path = Path(audio_path).resolve()
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Copy audio file to temp directory (to simplify Docker volume mounting)
            temp_audio = temp_dir_path / audio_path.name
            shutil.copy2(audio_path, temp_audio)
            
            # Build the Docker command
            docker_cmd = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "-v", f"{temp_dir_path}:/workspace",  # Mount temp directory
                self.docker_image,
                "/app/whisper.cpp/build/bin/whisper-cli",
                "-m", self.model_path,
                "-f", f"/workspace/{audio_path.name}",
                "-l", self.language,
                "--output-json" if output_format == "json" else f"--output-{output_format}",
                "-of", "/workspace/output"  # Output file prefix
            ]
            
            # Add any additional arguments
            if additional_args:
                docker_cmd.extend(additional_args)
            
            print(f"Running transcription with command: {' '.join(docker_cmd)}")
            
            try:
                # Run the Docker command
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Read the output file
                output_file = temp_dir_path / f"output.{output_format}"
                
                if output_format == "json":
                    with open(output_file, 'r') as f:
                        transcript_data = json.load(f)
                    
                    # Extract and format the transcript
                    return self._format_json_output(transcript_data)
                else:
                    with open(output_file, 'r') as f:
                        transcript_text = f.read()
                    
                    return {
                        "text": transcript_text,
                        "format": output_format,
                        "model": self.model_name,
                        "language": self.language
                    }
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Docker command failed: {e.stderr}"
                print(f"Error: {error_msg}")
                raise RuntimeError(error_msg)
            except FileNotFoundError as e:
                if "docker" in str(e):
                    raise RuntimeError("Docker not found. Please ensure Docker is installed and running.")
                raise
    
    def _format_json_output(self, transcript_data: Dict) -> Dict[str, Any]:
        """
        Format the JSON output from whisper-cpp into a standardized format
        
        Args:
            transcript_data: Raw JSON output from whisper-cpp
            
        Returns:
            Formatted transcript dictionary
        """
        # Extract segments and combine into full text
        segments = []
        full_text = []
        
        if "transcription" in transcript_data:
            for segment in transcript_data["transcription"]:
                segments.append({
                    "start": segment.get("offsets", {}).get("from", 0) / 1000.0,  # Convert ms to seconds
                    "end": segment.get("offsets", {}).get("to", 0) / 1000.0,
                    "text": segment.get("text", "").strip()
                })
                full_text.append(segment.get("text", "").strip())
        
        return {
            "text": " ".join(full_text),
            "segments": segments,
            "language": transcript_data.get("language", self.language),
            "model": self.model_name,
            "format": "json"
        }
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        output_format: str = "json"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            output_format: Output format for all transcriptions
            
        Returns:
            Dictionary mapping audio paths to their transcripts
        """
        results = {}
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\nTranscribing file {i}/{len(audio_paths)}: {audio_path}")
            try:
                results[audio_path] = self.transcribe(audio_path, output_format)
            except Exception as e:
                print(f"Failed to transcribe {audio_path}: {e}")
                results[audio_path] = {"error": str(e)}
        
        return results
    
    def check_docker_image(self) -> bool:
        """
        Check if the Docker image exists locally
        
        Returns:
            True if image exists, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    def build_docker_image(self, dockerfile_path: str = "metal/Dockerfile") -> bool:
        """
        Build the Docker image from the Dockerfile
        
        Args:
            dockerfile_path: Path to the Dockerfile
            
        Returns:
            True if build successful, False otherwise
        """
        try:
            print(f"Building Docker image {self.docker_image} from {dockerfile_path}...")
            subprocess.run(
                ["docker", "build", "-t", self.docker_image, "-f", dockerfile_path, "."],
                check=True
            )
            print("Docker image built successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to build Docker image: {e}")
            return False