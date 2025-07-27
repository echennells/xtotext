#!/usr/bin/env python3
"""Download specific episodes from Rogue Trader"""

import subprocess
import json
from pathlib import Path

# Episodes to download (mix of different types)
episodes = [
    {"id": "MgvoB_kGs1w", "title": "BITCOIN : Jul Price Update & Analysis"},
    {"id": "oq2RMsj613k", "title": "STOCK ANALYSIS BLITZ : Rio Tinto"},
    {"id": "nA27gON4AjA", "title": "STOCK ANALYSIS BLITZ : BHP Group"},
    {"id": "a8kG9NcmEe0", "title": "STOCK ANALYSIS UPDATE  - Natwest Group"},
    {"id": "s_RNlPve5Us", "title": "STOCK ANALYSIS UPDATE  - Rockhopper Exploration"}
]

output_dir = "downloads/rogue_trader"

for ep in episodes:
    video_url = f"https://youtube.com/watch?v={ep['id']}"
    print(f"\nDownloading: {ep['title']}")
    
    cmd = [
        'yt-dlp',
        '-x',  # Extract audio
        '--audio-format', 'mp3',
        '--audio-quality', '128K',
        '-o', f'{output_dir}/%(title)s_%(id)s.%(ext)s',
        video_url
    ]
    
    try:
        subprocess.run(cmd, check=True, timeout=60)  # 60 second timeout per video
        print(f"✓ Downloaded: {ep['title']}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"✗ Failed: {ep['title']} - {e}")

print("\nDownload complete!")