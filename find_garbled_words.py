#!/usr/bin/env python3
"""
Find garbled/nonsensical words in transcripts that might be speech-to-text errors
"""
import json
import re
from pathlib import Path
from collections import Counter
import string

def find_garbled_words(transcript_dir: str):
    """Find words that don't make sense in context"""
    
    # Common English words to skip (without system dictionary)
    common_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'would', 'her', 'she', 'he', 'it',
        'with', 'this', 'that', 'from', 'they', 'will', 'can', 'out', 'if', 'up', 'about', 'so',
        'said', 'what', 'its', 'who', 'get', 'has', 'him', 'how', 'man', 'back', 'now', 'way',
        'only', 'think', 'just', 'know', 'take', 'see', 'come', 'could', 'made', 'find', 'use',
        'than', 'been', 'call', 'first', 'may', 'water', 'oil', 'down', 'did', 'yeah', 'yes',
        'no', 'oh', 'ah', 'um', 'uh', 'okay', 'ok', 'well', 'like', 'right', 'mean', 'say',
        'go', 'going', 'gonna', 'got', 'getting', 'want', 'wanted', 'been', 'being', 'have',
        'having', 'had', 'make', 'making', 'made', 'take', 'taking', 'took', 'taken', 'give',
        'giving', 'gave', 'given', 'tell', 'telling', 'told', 'ask', 'asking', 'asked'
    }
    
    # Financial terms we know are valid
    financial_terms = {
        'btc', 'bitcoin', 'eth', 'ethereum', 'mstr', 'smlr', 'coin', 'stock', 'price', 'market',
        'trade', 'trading', 'trader', 'buy', 'sell', 'hold', 'pump', 'dump', 'moon', 'crypto',
        'defi', 'nft', 'blockchain', 'mining', 'miner', 'hash', 'wallet', 'exchange', 'dex',
        'yield', 'apy', 'apr', 'liquidity', 'swap', 'stake', 'staking', 'validator', 'node'
    }
    
    all_garbled = Counter()
    context_examples = {}
    
    transcript_files = list(Path(transcript_dir).glob("*.json"))
    print(f"Analyzing {len(transcript_files)} transcripts for garbled words...\n")
    
    for i, transcript_file in enumerate(transcript_files):
        print(f"[{i+1}/{len(transcript_files)}] {transcript_file.name}")
        
        try:
            with open(transcript_file, 'r') as f:
                data = json.load(f)
                text = data.get('text', '')
            
            # Find all words
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            
            for word in words:
                word_lower = word.lower()
                
                # Skip if too short or too long
                if len(word_lower) < 3 or len(word_lower) > 20:
                    continue
                
                # Skip if it's a known word
                if word_lower in common_words or word_lower in financial_terms:
                    continue
                
                # Skip if it's just repeated letters
                if len(set(word_lower)) < 3:  # e.g., "aaa", "haha"
                    continue
                
                # Check for patterns that suggest garbled text
                garbled_indicators = [
                    # Unusual consonant clusters
                    re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', word_lower),
                    # Too many vowels in a row
                    re.search(r'[aeiou]{4,}', word_lower),
                    # Alternating same letters
                    re.search(r'([a-z])\1{2,}', word_lower),
                    # No vowels at all (except common abbreviations)
                    len(word_lower) > 4 and not any(v in word_lower for v in 'aeiouy'),
                ]
                
                if any(garbled_indicators):
                    all_garbled[word_lower] += 1
                    
                    # Save context example
                    if word_lower not in context_examples and all_garbled[word_lower] == 1:
                        # Find context
                        pattern = r'\b' + re.escape(word) + r'\b'
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            context_examples[word_lower] = {
                                'context': text[start:end],
                                'episode': transcript_file.stem
                            }
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Also look for repeated nonsense patterns
    print("\n\nLooking for repeated patterns...")
    pattern_examples = find_repeated_patterns(transcript_dir)
    
    # Show results
    print("\n\n=== MOST COMMON GARBLED WORDS ===")
    for word, count in all_garbled.most_common(50):
        if count >= 3:  # Only show if it appears multiple times
            print(f"\n'{word}' - {count} occurrences")
            if word in context_examples:
                ctx = context_examples[word]
                print(f"  Episode: {ctx['episode']}")
                print(f"  Context: ...{ctx['context']}...")
    
    print("\n\n=== REPEATED NONSENSE PATTERNS ===")
    for pattern, examples in pattern_examples.items():
        if len(examples) >= 2:
            print(f"\nPattern: {pattern}")
            for ex in examples[:3]:
                print(f"  - {ex}")
    
    # Save results
    results = {
        'garbled_words': dict(all_garbled.most_common(100)),
        'context_examples': context_examples,
        'repeated_patterns': pattern_examples
    }
    
    output_path = "data/episodes/bitcoin_dive_bar_analysis/garbled_words_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")

def find_repeated_patterns(transcript_dir: str) -> dict:
    """Find repeated nonsense patterns like 'we call them we poor we call them we poor'"""
    patterns = {}
    
    for transcript_file in Path(transcript_dir).glob("*.json"):
        try:
            with open(transcript_file, 'r') as f:
                data = json.load(f)
                text = data.get('text', '').lower()
            
            # Look for repeated phrases (3+ words repeated)
            repeated_pattern = re.findall(r'(\b\w+\s+\w+\s+\w+\b)(?=.*\1)', text)
            
            for pattern in set(repeated_pattern):
                if pattern not in patterns:
                    patterns[pattern] = []
                
                # Find full context
                match = re.search(re.escape(pattern) + r'.*?' + re.escape(pattern), text)
                if match and len(match.group()) < 200:
                    patterns[pattern].append(match.group())
                    
        except Exception:
            continue
    
    return {k: v for k, v in patterns.items() if len(v) >= 2}

if __name__ == "__main__":
    find_garbled_words("data/episodes/bitcoin_dive_bar_analysis/transcripts")