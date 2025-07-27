#!/usr/bin/env python3
"""
Download episodes from Rogue Trader YouTube channel
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def get_channel_videos(channel_url, max_videos=10):
    """Get list of recent videos from channel"""
    print(f"Fetching video list from {channel_url}")
    
    # Use yt-dlp to get video info without downloading
    cmd = [
        'yt-dlp',
        '--flat-playlist',
        '--playlist-end', str(max_videos),
        '--print', '%(id)s|%(title)s|%(upload_date)s|%(duration)s',
        channel_url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        videos = []
        
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split('|')
                if len(parts) >= 4:
                    video_id, title, upload_date, duration = parts[:4]
                    videos.append({
                        'id': video_id,
                        'title': title,
                        'upload_date': upload_date,
                        'duration': int(duration) if duration.isdigit() else 0,
                        'url': f'https://youtube.com/watch?v={video_id}'
                    })
        
        return videos
    except subprocess.CalledProcessError as e:
        print(f"Error fetching videos: {e}")
        return []

def download_audio(video_url, output_dir):
    """Download audio from video"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # yt-dlp command to download audio
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'mp3',
        '--audio-quality', '128K',
        '-o', f'{output_dir}/%(title)s_%(id)s.%(ext)s',
        video_url
    ]
    
    try:
        print(f"Downloading: {video_url}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {video_url}: {e}")
        return False

def main():
    channel_url = "https://www.youtube.com/@roguetrader100/videos"
    output_dir = "downloads/rogue_trader"
    max_videos = 10
    
    print(f"=== Rogue Trader Episode Downloader ===")
    print(f"Channel: {channel_url}")
    print(f"Max videos: {max_videos}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Get video list
    videos = get_channel_videos(channel_url, max_videos)
    
    if not videos:
        print("No videos found!")
        return
    
    print(f"Found {len(videos)} videos:")
    for i, video in enumerate(videos, 1):
        duration_min = video['duration'] // 60
        print(f"{i}. {video['title']} ({duration_min} min)")
    
    # Create metadata file
    metadata_file = Path(output_dir) / "metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_file, 'w') as f:
        json.dump({
            'channel': 'Rogue Trader',
            'channel_url': channel_url,
            'download_date': datetime.now().isoformat(),
            'videos': videos
        }, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    
    # Download each video
    print(f"\nDownloading {len(videos)} videos...")
    successful = 0
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video['title']}")
        if download_audio(video['url'], output_dir):
            successful += 1
    
    print(f"\n=== Download Complete ===")
    print(f"Successfully downloaded: {successful}/{len(videos)} videos")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()