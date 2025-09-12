#!/usr/bin/env python3
"""
Quick demo: Extract key quotes with YouTube timestamps
Shows how we match quotes to timestamps and generate links
"""
import json
import re
from pathlib import Path


def demo_quote_extraction():
    """Demo showing how quote extraction works"""
    
    # Load one transcript as example
    transcript_file = Path("data/youtube_analysis/transcripts/bitcoin_Istanbul_Scaling_Bitcoin_Live_Day_1_gWWxDd3mhZc_022-058_transcript.json")
    
    with open(transcript_file, 'r') as f:
        data = json.load(f)
    
    text = data['text']
    segments = data['segments']
    
    print("=" * 70)
    print("DEMO: Quote Extraction with YouTube Timestamps")
    print("=" * 70)
    print(f"\nTranscript: {transcript_file.name}")
    print(f"Total segments: {len(segments)}")
    print(f"Duration: {segments[-1]['end']:.0f} seconds (~{segments[-1]['end']/60:.1f} minutes)")
    
    # Find key quotes using patterns
    print("\n" + "=" * 70)
    print("FINDING KEY QUOTES")
    print("=" * 70)
    
    # Pattern 1: Major realizations
    realization_pattern = r"(I realized[^.]+\.)"
    realizations = re.findall(realization_pattern, text, re.IGNORECASE)
    
    if realizations:
        print("\n🔴 REALIZATIONS:")
        for quote in realizations[:2]:
            # Find which segment contains this quote
            for segment in segments:
                if quote.lower()[:30] in segment['text'].lower():
                    time = int(segment['start'])
                    youtube_link = f"https://www.youtube.com/watch?v=gWWxDd3mhZc&t={time}"
                    print(f"\n  Quote: \"{quote[:100]}...\"")
                    print(f"  Time: {time//60}:{time%60:02d}")
                    print(f"  Link: {youtube_link}")
                    break
    
    # Pattern 2: Performance metrics
    metric_pattern = r"([^.]*(?:\d+x|\d+%|megabyte|kilobyte)[^.]+\.)"
    metrics = re.findall(metric_pattern, text, re.IGNORECASE)
    
    if metrics:
        print("\n🟡 PERFORMANCE METRICS:")
        for quote in metrics[:3]:
            if len(quote) > 30:  # Filter out short matches
                # Find segment
                for segment in segments:
                    if quote.lower()[:30] in segment['text'].lower():
                        time = int(segment['start'])
                        youtube_link = f"https://www.youtube.com/watch?v=gWWxDd3mhZc&t={time}"
                        print(f"\n  Quote: \"{quote[:100]}...\"")
                        print(f"  Time: {time//60}:{time%60:02d}")  
                        print(f"  Link: {youtube_link}")
                        break
    
    # Pattern 3: Key technical terms
    print("\n📝 KEY TECHNICAL CONCEPTS MENTIONED:")
    concepts = ["BitVM2", "BitVM3", "Garbled Circuits", "SNARK verifier", "Groth16"]
    
    for concept in concepts:
        # Find first mention
        pattern = re.compile(re.escape(concept), re.IGNORECASE)
        match = pattern.search(text)
        if match:
            # Find the segment
            for segment in segments:
                if concept.lower() in segment['text'].lower():
                    time = int(segment['start'])
                    youtube_link = f"https://www.youtube.com/watch?v=gWWxDd3mhZc&t={time}"
                    
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    print(f"\n  {concept}:")
                    print(f"    First mention at {time//60}:{time%60:02d}")
                    print(f"    Context: \"...{context}...\"")
                    print(f"    Link: {youtube_link}")
                    break
    
    print("\n" + "=" * 70)
    print("HOW IT WORKS:")
    print("=" * 70)
    print("1. We search for key phrases in the transcript text")
    print("2. Match each quote to its segment using the timestamp data")
    print("3. Generate YouTube links with &t=SECONDS parameter")
    print("4. Users can click links to jump directly to that moment!")
    print("=" * 70)


if __name__ == "__main__":
    demo_quote_extraction()