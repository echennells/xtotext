#!/usr/bin/env python3
"""
Find nonsensical patterns in transcripts using regex
"""
import json
import re
from pathlib import Path
from collections import Counter

def find_nonsense_patterns(transcript_dir: str):
    """Find repeated and garbled patterns"""
    
    all_patterns = {
        'repeated_phrases': Counter(),
        'garbled_words': Counter(),
        'repeated_words': Counter(),
        'broken_sentences': []
    }
    
    context_examples = {}
    
    transcript_files = list(Path(transcript_dir).glob("*.json"))
    print(f"Analyzing {len(transcript_files)} transcripts...\n")
    
    for i, transcript_file in enumerate(transcript_files):
        print(f"[{i+1}/{len(transcript_files)}] {transcript_file.name}")
        
        try:
            with open(transcript_file, 'r') as f:
                data = json.load(f)
                text = data.get('text', '')
            
            # Find repeated phrases (like "we call them we poor we call them we poor")
            # Look for 3+ word phrases that repeat immediately
            repeated = re.findall(r'(\b\w+(?:\s+\w+){2,4}\b)\s+\1', text, re.IGNORECASE)
            for phrase in repeated:
                if len(phrase) > 10:  # Skip very short phrases
                    all_patterns['repeated_phrases'][phrase.lower()] += 1
                    if phrase.lower() not in context_examples:
                        # Find full repetition
                        pattern = re.escape(phrase) + r'\s+' + re.escape(phrase)
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            start = max(0, match.start() - 20)
                            end = min(len(text), match.end() + 20)
                            context_examples[phrase.lower()] = {
                                'text': text[start:end],
                                'episode': transcript_file.stem
                            }
            
            # Find repeated single words (stuttering)
            stutters = re.findall(r'\b(\w+)\s+\1\s+\1\b', text, re.IGNORECASE)
            for word in stutters:
                all_patterns['repeated_words'][word.lower()] += 1
            
            # Find potential garbled words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            for word in words:
                word_lower = word.lower()
                
                # Check for unusual patterns
                if any([
                    # No vowels in long words
                    len(word_lower) > 5 and not any(v in word_lower for v in 'aeiouy'),
                    # Too many consonants in a row
                    re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', word_lower),
                    # Weird letter combinations
                    any(combo in word_lower for combo in ['xxx', 'qqq', 'zzz', 'vvv']),
                    # Multiple q's not followed by u
                    word_lower.count('q') > 1 and 'qu' not in word_lower,
                ]):
                    all_patterns['garbled_words'][word_lower] += 1
            
            # Find broken sentence patterns
            # Look for lowercase letters starting sentences after periods
            broken = re.findall(r'\.\s+[a-z]\w+', text)
            all_patterns['broken_sentences'].extend(broken[:5])  # Just first 5 per transcript
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Show results
    print("\n\n=== MOST COMMON REPEATED PHRASES ===")
    for phrase, count in all_patterns['repeated_phrases'].most_common(20):
        if count >= 2:
            print(f"\n'{phrase}' - repeated {count} times")
            if phrase in context_examples:
                ctx = context_examples[phrase]
                print(f"  Episode: {ctx['episode']}")
                print(f"  Example: ...{ctx['text']}...")
    
    print("\n\n=== MOST COMMON STUTTERED WORDS ===")
    for word, count in all_patterns['repeated_words'].most_common(20):
        print(f"'{word}' - stuttered {count} times")
    
    print("\n\n=== POTENTIAL GARBLED WORDS ===")
    for word, count in all_patterns['garbled_words'].most_common(30):
        if count >= 3:  # Only show if appears multiple times
            print(f"'{word}' - {count} occurrences")
    
    # Look for specific nonsense from Episode 1
    print("\n\n=== SPECIFIC NONSENSE PATTERNS ===")
    
    # From the Episode 1 example you mentioned
    test_file = Path(transcript_dir) / "Bitcoin Dive Bar EP 01 - Bitcoin All Time Highs_iuCuCG-4V7E_transcript.json"
    if test_file.exists():
        with open(test_file, 'r') as f:
            data = json.load(f)
            text = data.get('text', '')
        
        # Look for the "we call them we poor" pattern
        we_poor_pattern = re.search(r'we call them.*?we poor.*?we call them.*?we poor', text, re.IGNORECASE | re.DOTALL)
        if we_poor_pattern:
            print("\nFound 'we call them we poor' pattern:")
            start = max(0, we_poor_pattern.start() - 50)
            end = min(len(text), we_poor_pattern.end() + 50)
            print(f"Context: ...{text[start:end]}...")

if __name__ == "__main__":
    find_nonsense_patterns("data/episodes/bitcoin_dive_bar_analysis/transcripts")