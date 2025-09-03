"""
Utilities for sanitizing filenames
"""
import re
import unicodedata
from pathlib import Path


def sanitize_filename(filename: str, preserve_extension: bool = True) -> str:
    """
    Sanitize a filename by removing special characters and replacing spaces with underscores.
    Uses Python's built-in libraries for simplicity.
    
    Args:
        filename: The filename to sanitize
        preserve_extension: Whether to preserve the file extension
        
    Returns:
        Sanitized filename
    """
    if preserve_extension:
        path = Path(filename)
        name = path.stem
        ext = path.suffix
    else:
        name = filename
        ext = ""
    
    # Normalize unicode characters (convert things like full-width chars to ASCII equivalents)
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    # Remove/replace problematic characters for filesystems
    # Keep only alphanumeric, underscore, dash, and dot
    name = re.sub(r'[^\w\-.]', '_', name)
    
    # Clean up multiple underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores and dots
    name = name.strip('_.')
    
    # Ensure the filename is not empty
    if not name:
        name = 'unnamed'
    
    return name + ext