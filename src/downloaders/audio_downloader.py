import shutil
from pathlib import Path

from downloaders.x_downloader import XDownloader
from extractors.youtube_segment_extractor import YouTubeSegmentExtractor


def download_youtube_audio(video_id, download_dir, start_time=None, end_time=None):
    """
    Download YouTube video audio, optionally extracting a specific time range.

    Args:
        video_id: YouTube video ID
        download_dir: Directory to save the download
        start_time: Start time in HH:MM format (optional)
        end_time: End time in HH:MM format (optional)

    Returns:
        Path to the downloaded/extracted audio file
    """
    print(f"Downloading YouTube video: {video_id}", flush=True)

    if start_time and end_time:
        extractor = YouTubeSegmentExtractor(
            cache_dir="data/youtube_cache",
            output_dir=str(download_dir)
        )
        success, audio_path, metadata = extractor.extract_segment(
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            keep_full_audio=True
        )
        if not success:
            raise RuntimeError(f"Failed to extract segment from video {video_id}")
        print(f"✓ YouTube link saved: {metadata['youtube_timestamp_link']}")
        print(f"✓ Metadata saved with segment info")
        return audio_path

    # Check if we already have the audio file in cache
    cache_dir = Path("data/youtube_cache")
    existing_files = list(cache_dir.glob(f"*{video_id}*.mp3")) if cache_dir.exists() else []
    if existing_files:
        print(f"✓ Found existing audio file: {existing_files[0]}", flush=True)
        return existing_files[0]

    extractor = YouTubeSegmentExtractor(
        cache_dir="data/youtube_cache",
        output_dir=str(download_dir)
    )
    success, audio_path = extractor.download_full_audio(video_id, keep_cache=True)
    if not success:
        raise RuntimeError(f"Failed to download video {video_id}")
    return audio_path


def download_x_audio(x_id, download_dir, x_url=None):
    """Download X/Twitter video/space audio"""
    print(f"Downloading X/Twitter content: {x_id}")

    existing_files = list(Path("data").rglob(f"*{x_id}*.mp3")) + \
                     list(Path("data").rglob(f"*{x_id}*.m4a"))
    if existing_files:
        existing_file = existing_files[0]
        print(f"Found existing audio file: {existing_file}")
        download_dir.mkdir(parents=True, exist_ok=True)
        target_file = download_dir / existing_file.name
        shutil.copy2(existing_file, target_file)
        print(f"Copied to: {target_file}")
        return target_file

    downloader = XDownloader(str(download_dir), download_timeout=1800)
    if not x_url:
        x_url = f"https://x.com/i/status/{x_id}"
    audio_file = downloader.download_audio(x_url)
    if not audio_file:
        raise RuntimeError(f"Failed to download audio from X content {x_id}")
    if not audio_file.exists():
        raise RuntimeError(f"Downloaded audio file not found: {audio_file}")
    return audio_file


def download_direct_url(url, download_dir):
    """Download a direct media file URL using requests with progress display."""
    import urllib.request
    from urllib.parse import urlparse

    parsed = urlparse(url)
    filename = Path(parsed.path).name
    dest = download_dir / filename
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress

    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Download failed or produced empty file: {dest}")

    print(f"✓ Downloaded: {dest}")
    return dest
