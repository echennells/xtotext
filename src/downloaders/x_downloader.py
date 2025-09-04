#!/usr/bin/env python3
"""
X (Twitter) Media Downloader - Extract audio/video from X posts
"""
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging


class XDownloader:
    """Download media (video/audio) from X/Twitter posts"""
    
    def __init__(self, output_dir: str = "data/x_downloads", download_timeout: int = 3600):
        """
        Initialize X downloader
        
        Args:
            output_dir: Directory to save downloaded media
            download_timeout: Timeout in seconds for download (default: 3600 = 60 minutes)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / ".x_download_state.json"
        self.state = self._load_state()
        self.logger = self._setup_logger()
        self.download_timeout = download_timeout
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the downloader"""
        logger = logging.getLogger('XDownloader')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_state(self) -> Dict:
        """Load download state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "downloaded": {},  # post_id: {username, text, file, date, media_type}
            "failed": {},      # post_id: {url, error, attempts}
        }
    
    def _save_state(self):
        """Save download state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _extract_post_id(self, url: str) -> Optional[str]:
        """Extract post ID from X/Twitter URL"""
        # Match patterns like:
        # https://x.com/username/status/1234567890
        # https://twitter.com/username/status/1234567890
        pattern = r'(?:x\.com|twitter\.com)/\w+/status/(\d+)'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    
    def _get_post_metadata(self, url: str) -> Dict:
        """Get metadata about the X post"""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Could not get metadata: {e}")
        
        return {}
    
    def download_media(self, url: str, extract_audio: bool = True) -> Optional[Path]:
        """
        Download media from X/Twitter post
        
        Args:
            url: X/Twitter post URL
            extract_audio: If True, extract audio from video; if False, keep original format
            
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        post_id = self._extract_post_id(url)
        
        if not post_id:
            self.logger.error(f"Could not extract post ID from URL: {url}")
            return None
        
        # Check if already downloaded
        if post_id in self.state["downloaded"]:
            existing_file = Path(self.state["downloaded"][post_id]["file"])
            if existing_file.exists():
                self.logger.info(f"Already downloaded: {existing_file}")
                return existing_file
        
        self.logger.info(f"Downloading from: {url}")
        self.logger.info(f"Post ID: {post_id}")
        
        # Get metadata first
        metadata = self._get_post_metadata(url)
        username = metadata.get("uploader", "unknown")
        description = metadata.get("description", "")[:100]  # First 100 chars
        
        # Prepare download command
        output_template = str(self.output_dir / f"{username}_{post_id}_%(title)s.%(ext)s")
        
        if extract_audio:
            cmd = [
                "yt-dlp",
                url,
                "--extract-audio",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", output_template,
                "--no-playlist",
                "--no-warnings"
            ]
            expected_ext = "mp3"
        else:
            cmd = [
                "yt-dlp",
                url,
                "-o", output_template,
                "--no-playlist",
                "--no-warnings"
            ]
            expected_ext = "mp4"  # Most common for X videos
        
        try:
            self.logger.info("Starting download...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.download_timeout)
            
            if result.returncode == 0:
                # Find the downloaded file
                pattern = f"{username}_{post_id}_*"
                downloaded_files = list(self.output_dir.glob(pattern))
                
                if downloaded_files:
                    downloaded_file = downloaded_files[0]  # Take the most recent
                    
                    # Update state
                    self.state["downloaded"][post_id] = {
                        "username": username,
                        "text": description,
                        "file": str(downloaded_file),
                        "date": datetime.now().isoformat(),
                        "media_type": "audio" if extract_audio else "video",
                        "url": url,
                        "size_mb": downloaded_file.stat().st_size / (1024 * 1024)
                    }
                    
                    # Remove from failed if it was there
                    if post_id in self.state["failed"]:
                        del self.state["failed"][post_id]
                    
                    self._save_state()
                    
                    self.logger.info(f"✓ Download successful: {downloaded_file.name}")
                    self.logger.info(f"  Size: {self.state['downloaded'][post_id]['size_mb']:.2f} MB")
                    
                    return downloaded_file
                else:
                    raise Exception("Download seemed successful but file not found")
                    
            else:
                error_msg = result.stderr or "Unknown error"
                self.logger.error(f"Download failed: {error_msg}")
                
                # Update failed state
                if post_id not in self.state["failed"]:
                    self.state["failed"][post_id] = {
                        "url": url,
                        "error": error_msg[:500],  # Truncate long errors
                        "attempts": 1,
                        "last_attempt": datetime.now().isoformat()
                    }
                else:
                    self.state["failed"][post_id]["attempts"] += 1
                    self.state["failed"][post_id]["last_attempt"] = datetime.now().isoformat()
                    self.state["failed"][post_id]["error"] = error_msg[:500]
                
                self._save_state()
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Download timed out after {self.download_timeout} seconds")
            self.state["failed"][post_id] = {
                "url": url,
                "error": "Timeout",
                "attempts": self.state.get("failed", {}).get(post_id, {}).get("attempts", 0) + 1,
                "last_attempt": datetime.now().isoformat()
            }
            self._save_state()
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.state["failed"][post_id] = {
                "url": url,
                "error": str(e)[:500],
                "attempts": self.state.get("failed", {}).get(post_id, {}).get("attempts", 0) + 1,
                "last_attempt": datetime.now().isoformat()
            }
            self._save_state()
        
        return None
    
    def download_audio(self, url: str) -> Optional[Path]:
        """
        Convenience method to download and extract audio
        
        Args:
            url: X/Twitter post URL
            
        Returns:
            Path to downloaded audio file if successful
        """
        return self.download_media(url, extract_audio=True)
    
    def download_video(self, url: str) -> Optional[Path]:
        """
        Convenience method to download video in original format
        
        Args:
            url: X/Twitter post URL
            
        Returns:
            Path to downloaded video file if successful
        """
        return self.download_media(url, extract_audio=False)
    
    def get_download_status(self) -> Dict:
        """Get summary of download status"""
        return {
            "total_downloaded": len(self.state["downloaded"]),
            "total_failed": len(self.state["failed"]),
            "total_size_mb": sum(
                item.get("size_mb", 0) 
                for item in self.state["downloaded"].values()
            ),
            "recent_downloads": sorted(
                [
                    {
                        "post_id": post_id,
                        "username": info["username"],
                        "date": info["date"],
                        "file": Path(info["file"]).name
                    }
                    for post_id, info in self.state["downloaded"].items()
                ],
                key=lambda x: x["date"],
                reverse=True
            )[:5]
        }
    
    def retry_failed(self, max_attempts: int = 3) -> Dict[str, bool]:
        """
        Retry failed downloads
        
        Args:
            max_attempts: Maximum number of attempts per URL
            
        Returns:
            Dict of post_id: success for retried downloads
        """
        results = {}
        
        for post_id, info in list(self.state["failed"].items()):
            if info["attempts"] < max_attempts:
                self.logger.info(f"Retrying {post_id} (attempt {info['attempts'] + 1}/{max_attempts})")
                result = self.download_media(info["url"])
                results[post_id] = result is not None
        
        return results


def main():
    """Example usage"""
    downloader = XDownloader()
    
    # Test with the provided URL
    test_url = "https://x.com/david_seroy/status/1953471306894299410"
    
    print("="*80)
    print("X/Twitter Media Downloader")
    print("="*80)
    
    # Try to download audio
    audio_file = downloader.download_audio(test_url)
    
    if audio_file:
        print(f"\n✓ Success! Audio saved to: {audio_file}")
    else:
        print("\n✗ Failed to download audio")
        print("\nTrying to download as video instead...")
        
        video_file = downloader.download_video(test_url)
        if video_file:
            print(f"\n✓ Success! Video saved to: {video_file}")
        else:
            print("\n✗ Failed to download media")
    
    # Show status
    print("\n" + "="*80)
    print("Download Status:")
    print("="*80)
    status = downloader.get_download_status()
    print(f"Total downloaded: {status['total_downloaded']}")
    print(f"Total failed: {status['total_failed']}")
    print(f"Total size: {status['total_size_mb']:.2f} MB")
    
    if status['recent_downloads']:
        print("\nRecent downloads:")
        for dl in status['recent_downloads']:
            print(f"  - {dl['username']}: {dl['file']}")


if __name__ == "__main__":
    main()