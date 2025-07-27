#!/usr/bin/env python3
"""
Use LLM to find nonsensical/garbled text in transcripts
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.llm_client import OpenAIClient
import json
import random

def analyze_transcript_for_nonsense(transcript_path: str, sample_size: int = 10):
    """Use LLM to find nonsensical text"""
    
    client = OpenAIClient()
    client.model = "gpt-4-turbo"
    
    with open(transcript_path, 'r') as f:
        data = json.load(f)
        text = data.get('text', '')
    
    # Take random samples from the transcript
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Randomly sample chunks
    sample_chunks = random.sample(chunks, min(sample_size, len(chunks)))
    
    all_nonsense = []
    
    print(f"\nAnalyzing {len(sample_chunks)} samples from {Path(transcript_path).name}")
    
    for i, chunk in enumerate(sample_chunks):
        prompt = """Analyze this podcast transcript excerpt for nonsensical or garbled text that seems like speech-to-text errors.

Look for:
1. Repeated phrases that don't make sense (e.g., "we call them we poor we call them we poor")
2. Garbled words that aren't real words or names
3. Sentences that are grammatically broken in ways that suggest transcription errors
4. Words that seem like misheard audio (e.g., "adjunive horse" instead of something else)

Return a JSON array of problems found:
[
  {
    "type": "repeated_nonsense|garbled_word|broken_grammar|misheard",
    "text": "the problematic text",
    "likely_meaning": "what it probably should be (if you can guess)",
    "context": "surrounding text for context"
  }
]

If no problems found, return empty array [].

Transcript excerpt:
""" + chunk

        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a transcript quality analyzer."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response['choices'][0]['message']['content']
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            
            problems = json.loads(content.strip())
            all_nonsense.extend(problems)
            
            if problems:
                print(f"  Sample {i+1}: Found {len(problems)} issues")
            
        except Exception as e:
            print(f"  Error in sample {i+1}: {e}")
            continue
    
    return all_nonsense

def main():
    transcript_dir = "data/episodes/bitcoin_dive_bar_analysis/transcripts"
    
    # Test on a few transcripts
    test_files = [
        "Bitcoin Dive Bar EP 5 - Bitcoin 120k  Guests BTC SMLR MSTR_vuA8LrC_z3s_transcript.json",
        "Bitcoin Dive Bar EP 9 - Tim B  Be Scarce - Back to 120000 BTC SMLR MSTR_2AtpFR-fCrU_transcript.json",
        "Bitcoin Dive Bar EP 01 - Bitcoin All Time Highs_iuCuCG-4V7E_transcript.json"
    ]
    
    all_findings = []
    
    for filename in test_files:
        filepath = Path(transcript_dir) / filename
        if filepath.exists():
            findings = analyze_transcript_for_nonsense(str(filepath), sample_size=5)
            all_findings.extend(findings)
    
    # Group by type
    print("\n\n=== NONSENSE TEXT ANALYSIS ===")
    
    by_type = {}
    for finding in all_findings:
        ftype = finding.get('type', 'unknown')
        if ftype not in by_type:
            by_type[ftype] = []
        by_type[ftype].append(finding)
    
    for ftype, items in by_type.items():
        print(f"\n{ftype.upper()} ({len(items)} found):")
        for item in items[:5]:  # Show first 5
            print(f"\n  Text: '{item['text']}'")
            if item.get('likely_meaning'):
                print(f"  Likely: '{item['likely_meaning']}'")
            print(f"  Context: ...{item.get('context', '')[:100]}...")
    
    # Save results
    output = {
        'findings': all_findings,
        'summary': {k: len(v) for k, v in by_type.items()}
    }
    
    output_path = "data/episodes/bitcoin_dive_bar_analysis/nonsense_text_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nFull results saved to: {output_path}")

if __name__ == "__main__":
    main()