#!/usr/bin/env python3
"""
Detect mentions of Eric Chennells and related terms in transcripts
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from extractors.mention_detector import MentionDetector


def main():
    """Run mention detection on Bitcoin Dive Bar transcripts"""
    
    # Set up paths
    transcript_dir = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts")
    output_dir = Path("data/episodes/bitcoin_dive_bar_analysis/mentions")
    
    if not transcript_dir.exists():
        print(f"Error: Transcript directory not found: {transcript_dir}")
        return
    
    print("=" * 80)
    print("MENTION DETECTION - Looking for Eric Chennells & Related Terms")
    print("=" * 80)
    
    # Initialize detector with custom search terms
    search_terms = [
        # Eric variations
        "eric chennells",
        "eric chanel",  # Common misspelling
        "eric channel",  # Common misspelling
        "lightning eric",
        "eric from podcast predictions",
        
        # Podcast/project names
        "podcast predictions",
        "prediction tracker",
        "predictiontracker",
        
        # Related terms
        "prediction tracking",
        "bitcoin predictions podcast",
        "predictions podcast",
        
        # Social media handles (if mentioned)
        "@echennells",
        "echennells"
    ]
    
    detector = MentionDetector(search_terms=search_terms)
    
    # Process all transcripts
    print(f"\nSearching transcripts in: {transcript_dir}")
    print(f"Output will be saved to: {output_dir}")
    print("\nSearch terms:")
    for term in search_terms:
        print(f"  - {term}")
    print()
    
    # Run detection
    summary = detector.process_batch(transcript_dir, output_dir)
    
    # Generate report
    report_path = output_dir / "mention_report.txt"
    detector.generate_report(summary, report_path)
    
    # Print quick summary
    if summary['total_mentions'] > 0:
        print(f"\n✓ Found {summary['total_mentions']} mentions across {summary['files_with_mentions']} files!")
        print(f"✓ Detailed reports saved to: {output_dir}")
        print(f"✓ Summary report: {report_path}")
        
        # Show a few examples
        print("\nSample mentions:")
        mention_files = list(output_dir.glob("*_mentions.json"))
        for mention_file in mention_files[:3]:  # Show first 3 files
            import json
            with open(mention_file, 'r') as f:
                data = json.load(f)
            
            if data['mentions']:
                print(f"\nFrom {data['transcript_file']}:")
                for mention in data['mentions'][:2]:  # Show first 2 mentions
                    print(f"  - {mention['term']} (timestamp: {mention['timestamp']:.1f}s)")
                    print(f"    {mention['context'][:100]}...")
    else:
        print("\n✗ No mentions found in any transcripts")
    
    return summary


if __name__ == "__main__":
    main()