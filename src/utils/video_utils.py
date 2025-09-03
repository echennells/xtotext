"""
Utility functions for video processing
"""
import subprocess
from typing import Optional, Tuple, List


def get_video_duration(url: str) -> int:
    """
    Get video duration in seconds using yt-dlp
    
    Args:
        url: YouTube video URL
        
    Returns:
        Duration in seconds, 0 if unable to fetch
    """
    try:
        cmd = ["yt-dlp", "--print", "duration", "--quiet", "--no-warnings", url]
        # Add 30 second timeout per video
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        duration_str = result.stdout.strip().split('\n')[0]  # Get first line
        return int(duration_str) if duration_str.isdigit() else 0
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        print(f"    (Failed to get duration)")
        return 0


def get_video_metadata(url: str) -> dict:
    """
    Get comprehensive video metadata
    
    Args:
        url: YouTube video URL
        
    Returns:
        Dictionary with title, duration, upload_date, etc.
    """
    try:
        cmd = [
            "yt-dlp",
            "--print", '%(id)s|%(title)s|%(duration)s|%(upload_date)s|%(view_count)s',
            "--quiet",
            "--no-warnings",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split('|')
        
        if len(parts) >= 3:
            return {
                'id': parts[0],
                'title': parts[1],
                'duration': int(parts[2]) if parts[2].isdigit() else 0,
                'upload_date': parts[3] if len(parts) > 3 else None,
                'view_count': int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
            }
    except:
        pass
    
    return {}


def filter_by_duration(videos: List[Tuple], min_duration_seconds: int = 1800, db=None) -> List[Tuple]:
    """
    Filter videos by minimum duration, checking database first
    
    Args:
        videos: List of (video_id, title, url) tuples
        min_duration_seconds: Minimum duration in seconds (default 30 minutes)
        db: Optional database instance to check for known videos
        
    Returns:
        List of videos that meet duration requirement with duration added
    """
    filtered = []
    min_duration_minutes = min_duration_seconds / 60
    
    # Count how many we actually need to check
    to_check = []
    for video_id, title, url in videos:
        if db:
            is_known, status = db.is_video_known(video_id, 'youtube')
            if is_known:
                print(f"  ⊖ Already {status}: {title[:60]}...")
                if status == 'processed':
                    # Include processed videos in the filtered list (they're full episodes)
                    video_entry = db.get_video(f"youtube_{video_id}")
                    if video_entry:
                        duration = video_entry.get('processing_stats', {}).get('duration_seconds', 0)
                        if duration >= min_duration_seconds:
                            filtered.append((video_id, title, url, duration))
                continue
        to_check.append((video_id, title, url))
    
    if not to_check:
        print("\nAll videos already known - nothing new to check")
        return filtered
    
    print(f"\nChecking {len(to_check)} new videos (minimum: {min_duration_minutes:.0f} minutes)...")
    print(f"This will take ~{len(to_check) * 5} seconds ({len(to_check)} videos × ~5 sec each)")
    
    for i, (video_id, title, url) in enumerate(to_check, 1):
        print(f"\n[{i}/{len(to_check)}] Checking: {title[:60]}...")
        
        # Get actual duration for this video
        duration = get_video_duration(url)
        duration_mins = duration / 60 if duration > 0 else 0
        
        if duration >= min_duration_seconds:
            filtered.append((video_id, title, url, duration))
            print(f"  ✓ FULL EPISODE ({duration_mins:.1f} minutes)")
        else:
            print(f"  ✗ SKIP - too short ({duration_mins:.1f} minutes)")
            # Log to database as skipped
            if db:
                db.add_skipped_video(
                    video_id=video_id,
                    title=title,
                    url=url,
                    reason='too_short',
                    duration_seconds=duration,
                    platform='youtube'
                )
    
    print(f"\nFound {len(filtered)} full episodes (including known) out of {len(videos)} total")
    return filtered


def is_full_episode(url: str, min_duration_minutes: int = 30) -> bool:
    """
    Check if a video is a full episode based on duration
    
    Args:
        url: YouTube video URL
        min_duration_minutes: Minimum duration in minutes to be considered full episode
        
    Returns:
        True if video is longer than minimum duration
    """
    duration = get_video_duration(url)
    return duration >= (min_duration_minutes * 60)