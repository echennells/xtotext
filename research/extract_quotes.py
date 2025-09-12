#!/usr/bin/env python3
"""
Extract key quotes with YouTube timestamp links
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from research.quote_extractor import QuoteExtractor


def main():
    print("=" * 70)
    print("QUOTE EXTRACTOR - Bitcoin++ Istanbul Conference")
    print("=" * 70)
    
    # Initialize extractor
    extractor = QuoteExtractor()
    
    # Extract quotes from all transcripts
    print("\nExtracting key quotes from all transcripts...")
    quotes = extractor.extract_all_quotes()
    
    print(f"\nTotal quotes found: {len(quotes)}")
    
    # Show top critical quotes
    critical = [q for q in quotes if q.importance == 'critical']
    if critical:
        print(f"\n🔴 CRITICAL INSIGHTS ({len(critical)}):")
        print("-" * 40)
        for q in critical[:3]:
            print(f"\n📍 {q.formatted_time} - {q.topic}")
            print(f"   \"{q.text[:100]}...\"")
            print(f"   🔗 {q.youtube_link}")
    
    # Generate reports
    output_dir = Path("research/outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Markdown report
    print("\nGenerating markdown report...")
    report = extractor.generate_quote_report()
    report_file = output_dir / "key_quotes_with_timestamps.md"
    report_file.write_text(report)
    print(f"  Saved to: {report_file}")
    
    # JSON export
    print("\nExporting JSON with all metadata...")
    json_file = output_dir / "quotes_with_timestamps.json"
    extractor.export_quotes_json(json_file)
    
    # Example of direct links
    print("\n" + "=" * 70)
    print("EXAMPLE YOUTUBE LINKS WITH TIMESTAMPS:")
    print("=" * 70)
    
    for q in quotes[:5]:
        print(f"\n{q.topic.upper()}")
        print(f"  Quote: \"{q.text[:80]}...\"")
        print(f"  Link: {q.youtube_link}")
        print(f"  Time: {q.formatted_time}")
    
    print("\n" + "=" * 70)
    print("You can now:")
    print("1. Open key_quotes_with_timestamps.md for a readable report")
    print("2. Click any timestamp to jump directly to that moment in the video")
    print("3. Use quotes_with_timestamps.json for further processing")
    print("=" * 70)


if __name__ == "__main__":
    main()