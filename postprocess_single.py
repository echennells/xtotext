#!/usr/bin/env python3
"""
Run post-processing on a single transcript file
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from processors.claude_transcript_postprocessor import postprocess_transcript_claude

def main():
    if len(sys.argv) != 2:
        print("Usage: python postprocess_single.py <transcript_file>")
        sys.exit(1)

    transcript_path = Path(sys.argv[1])

    if not transcript_path.exists():
        print(f"Error: File not found: {transcript_path}")
        sys.exit(1)

    print(f"Post-processing: {transcript_path}")

    try:
        output_path = postprocess_transcript_claude(transcript_path)
        print(f"✓ Successfully post-processed to: {output_path}")
    except Exception as e:
        print(f"✗ Error during post-processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()