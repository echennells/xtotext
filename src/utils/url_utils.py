import re
from pathlib import Path


def parse_video_url(url: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse a video URL and return (platform, video_id, canonical_url).

    Handles YouTube (including /live/ and /broadcasts/), X/Twitter (including
    Spaces and Broadcasts), and direct media file URLs.

    Returns (None, None, None) if the URL is not recognized.
    """
    # YouTube patterns
    youtube_patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/live/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$'  # Just the video ID
    ]

    # X/Twitter patterns
    x_patterns = [
        r'(?:twitter\.com|x\.com)/\w+/status/(\d+)',
        r'(?:twitter\.com|x\.com)/i/spaces/([a-zA-Z0-9]+)',  # Spaces pattern
        r'(?:twitter\.com|x\.com)/i/broadcasts/([a-zA-Z0-9]+)',  # Broadcasts pattern
        r'^(\d{15,20})$'  # Just the tweet ID
    ]

    # Check YouTube
    for pattern in youtube_patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            return 'youtube', video_id, f"https://www.youtube.com/watch?v={video_id}"

    # Check X/Twitter
    for pattern in x_patterns:
        match = re.search(pattern, url)
        if match:
            id_value = match.group(1)
            if '/spaces/' in url:
                return 'x', id_value, f"https://x.com/i/spaces/{id_value}"
            elif '/broadcasts/' in url:
                return 'x', id_value, f"https://x.com/i/broadcasts/{id_value}"
            else:
                return 'x', id_value, f"https://x.com/i/status/{id_value}"

    # Check for direct media file URLs
    media_extensions = r'\.(mp4|mp3|wav|m4a|ogg|webm|flac|aac|opus)(\?.*)?$'
    if re.search(media_extensions, url, re.IGNORECASE):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        filename = Path(parsed.path).stem
        video_id = re.sub(r'[^\w-]', '_', filename)
        return 'direct', video_id, url

    return None, None, None
