"""
YouTube Segment Extractor Module

Handles downloading YouTube videos and extracting specific time segments,
with metadata tracking and efficient caching.
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class YouTubeSegmentExtractor:
    """Extract specific time segments from YouTube videos with metadata tracking."""
    
    def __init__(self, cache_dir: str = "data/youtube_cache", output_dir: str = "data/youtube_segments"):
        """
        Initialize the YouTube segment extractor.
        
        Args:
            cache_dir: Directory to cache full downloads
            output_dir: Directory for extracted segments
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_time_to_seconds(self, time_str: str) -> int:
        """
        Parse time string to seconds.
        
        Args:
            time_str: Time in HH:MM format (e.g., "4:40" = 4 hours 40 minutes)
            
        Returns:
            Total seconds
        """
        parts = time_str.split(':')
        if len(parts) != 2:
            raise ValueError(f"Time must be in HH:MM format, got: {time_str}")
        
        hours, minutes = map(int, parts)
        return hours * 3600 + minutes * 60
    
    def format_time_for_display(self, seconds: int) -> str:
        """Format seconds to HH:MM:SS for display."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_cached_audio(self, video_id: str) -> Optional[Path]:
        """
        Check if we have a cached full audio file for this video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Path to cached audio file if exists, None otherwise
        """
        # Look for any MP3 file with this video ID in cache
        pattern = f"*{video_id}*.mp3"
        cached_files = list(self.cache_dir.glob(pattern))
        
        if cached_files:
            # Return the most recent one
            cached_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return cached_files[0]
        
        return None
    
    def download_full_audio(self, video_id: str, keep_cache: bool = True) -> Tuple[bool, Optional[Path]]:
        """
        Download full audio from YouTube video.
        
        Args:
            video_id: YouTube video ID
            keep_cache: Whether to keep the full audio cached
            
        Returns:
            Tuple of (success, filepath)
        """
        # Check cache first
        cached = self.get_cached_audio(video_id)
        if cached:
            logger.info(f"Using cached audio: {cached.name}")
            return True, cached
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Get video title for filename
        cmd = ["yt-dlp", "--get-title", url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Failed to get video title: {result.stderr}")
            title = f"video_{video_id}"
        else:
            title = result.stdout.strip()
            # Sanitize title for filename
            title = "".join(c for c in title if c.isalnum() or c in " -_").rstrip()
        
        # Download audio
        output_path = self.cache_dir / f"{title}_{video_id}.mp3"
        
        download_cmd = [
            "yt-dlp",
            url,
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--audio-quality", "0",  # Best quality
            "-o", str(output_path.with_suffix('.%(ext)s')),
            "--quiet",
            "--no-warnings"
        ]
        
        logger.info(f"Downloading audio from {url}...")
        start_time = datetime.now()
        result = subprocess.run(download_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Download failed: {result.stderr}")
            return False, None
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Find the downloaded file
        if not output_path.exists():
            # yt-dlp might have added something to the filename
            mp3_files = list(self.cache_dir.glob(f"*{video_id}*.mp3"))
            if mp3_files:
                output_path = mp3_files[0]
            else:
                logger.error("Downloaded file not found")
                return False, None
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {title} ({file_size_mb:.1f} MB in {elapsed:.1f}s)")
        
        return True, output_path
    
    def extract_segment(
        self,
        video_id: str,
        start_time: str,
        end_time: str,
        keep_full_audio: bool = False
    ) -> Tuple[bool, Optional[Path], Dict]:
        """
        Extract a time segment from a YouTube video.
        
        Args:
            video_id: YouTube video ID
            start_time: Start time in HH:MM format
            end_time: End time in HH:MM format
            keep_full_audio: Whether to keep the full audio cached after extraction
            
        Returns:
            Tuple of (success, segment_path, metadata_dict)
        """
        # Parse times
        start_seconds = self.parse_time_to_seconds(start_time)
        end_seconds = self.parse_time_to_seconds(end_time)
        duration = end_seconds - start_seconds
        
        if duration <= 0:
            logger.error(f"Invalid time range: {start_time} to {end_time}")
            return False, None, {}
        
        logger.info(f"Extracting segment from {start_time} to {end_time} ({duration//60} minutes)")
        
        # Download full audio
        success, full_audio_path = self.download_full_audio(video_id, keep_cache=keep_full_audio)
        if not success or not full_audio_path:
            return False, None, {}
        
        # Create output filename with time range
        safe_start = start_time.replace(':', '')
        safe_end = end_time.replace(':', '')
        segment_name = f"{full_audio_path.stem}_{safe_start}-{safe_end}.mp3"
        segment_path = self.output_dir / segment_name
        
        # Extract segment using ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(full_audio_path),
            "-ss", str(start_seconds),  # Start time in seconds
            "-t", str(duration),        # Duration in seconds
            "-acodec", "mp3",
            "-ab", "192k",
            str(segment_path),
            "-y"  # Overwrite if exists
        ]
        
        logger.info(f"Extracting segment with ffmpeg...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg extraction failed: {result.stderr}")
            return False, None, {}
        
        # Create metadata
        metadata = {
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "start_time": start_time,
            "end_time": end_time,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": duration,
            "duration_formatted": f"{duration//60} minutes {duration%60} seconds",
            "segment_file": segment_path.name,
            "full_audio_cached": keep_full_audio,
            "extracted_at": datetime.now().isoformat(),
            "youtube_timestamp_link": f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
        }
        
        # Save metadata to JSON file
        metadata_path = segment_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save a simple text file with the link and timestamp
        link_path = segment_path.with_suffix('.link.txt')
        with open(link_path, 'w') as f:
            f.write(f"YouTube Video: {metadata['video_url']}\n")
            f.write(f"Segment: {start_time} to {end_time}\n")
            f.write(f"Direct link to start time: {metadata['youtube_timestamp_link']}\n")
            f.write(f"Duration: {metadata['duration_formatted']}\n")
            f.write(f"Extracted: {metadata['extracted_at']}\n")
        
        # Clean up full audio if not keeping cache
        if not keep_full_audio and full_audio_path:
            logger.info("Removing cached full audio...")
            full_audio_path.unlink()
        
        segment_size_mb = segment_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Segment extracted: {segment_path.name} ({segment_size_mb:.1f} MB)")
        logger.info(f"✓ Metadata saved: {metadata_path.name}")
        logger.info(f"✓ Link saved: {link_path.name}")
        
        return True, segment_path, metadata
    
    def extract_multiple_segments(
        self,
        video_id: str,
        segments: list,
        keep_full_audio: bool = True
    ) -> list:
        """
        Extract multiple segments from the same video efficiently.
        
        Args:
            video_id: YouTube video ID
            segments: List of (start_time, end_time) tuples
            keep_full_audio: Whether to keep the full audio cached
            
        Returns:
            List of results for each segment
        """
        results = []
        
        # Download once, extract many
        for i, (start, end) in enumerate(segments):
            # Keep cache for all but the last segment
            keep_cache = keep_full_audio or (i < len(segments) - 1)
            success, path, metadata = self.extract_segment(
                video_id, start, end, keep_full_audio=keep_cache
            )
            results.append({
                "success": success,
                "path": path,
                "metadata": metadata
            })
        
        return results