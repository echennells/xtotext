import os


def get_vast_api_key() -> str | None:
    """Get Vast.ai API key from environment or config file."""
    key = os.getenv("VAST_API_KEY")
    if not key:
        try:
            from config.config import VAST_API_KEY
            key = VAST_API_KEY
        except ImportError:
            pass
    return key or None


def get_openrouter_api_key() -> str | None:
    """Get OpenRouter API key from environment or config file."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        try:
            from config.config import OPENROUTER_API_KEY
            key = OPENROUTER_API_KEY
        except ImportError:
            pass
    return key or None
