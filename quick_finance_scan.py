#!/usr/bin/env python3
"""
Quick scan for potential finance terms across all transcripts
Focuses on finding slang/misspellings without full LLM analysis
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

def scan_transcripts(transcript_dir: str):
    """Quick scan for potential finance terms"""
    
    # Known good mappings from Episode 5
    known_slang = {
        'misty': 'MSTR',
        'mst': 'MSTR', 
        'msty': 'MSTR',
        'similar': 'SMLR',
        'sampler': 'SMLR',
        'summer': 'SMLR',
        'sailor': 'Michael Saylor',
        'big coin': 'Bitcoin',
        'meta planet': 'MetaPlanet',
        'james street': 'Jane Street',
        'divi': 'dividend'
    }
    
    # Financial context patterns
    financial_patterns = [
        r'going to (\d+)',
        r'hits (\d+)',
        r'target (\d+)',
        r'headed to (\d+)',
        r'(\w+) to \$?(\d+)',
        r'(\w+) is (\d+)',
    ]
    
    all_findings = defaultdict(list)
    
    transcript_files = list(Path(transcript_dir).glob("*.json"))
    print(f"Scanning {len(transcript_files)} transcripts...")
    
    for i, transcript_file in enumerate(transcript_files):
        print(f"\n[{i+1}/{len(transcript_files)}] {transcript_file.name}")
        
        try:
            with open(transcript_file, 'r') as f:
                data = json.load(f)
                text = data.get('text', '').lower()
            
            # Look for known slang terms
            for slang, proper in known_slang.items():
                if slang in text:
                    count = text.count(slang)
                    if count > 2:  # Only if it appears multiple times
                        # Get a sample context
                        idx = text.find(slang)
                        context = text[max(0, idx-50):idx+50]
                        all_findings[proper].append({
                            'term': slang,
                            'count': count,
                            'episode': transcript_file.stem,
                            'context': context
                        })
                        print(f"  Found '{slang}' -> {proper} ({count} times)")
            
            # Look for potential new terms near price mentions
            for pattern in financial_patterns:
                matches = re.findall(pattern, text)
                for match in matches[:5]:  # First 5 matches
                    if isinstance(match, tuple):
                        term = match[0]
                        if len(term) > 2 and len(term) < 20:
                            # Check if it's not a common word
                            if term not in ['the', 'and', 'for', 'this', 'that', 'will', 'with']:
                                all_findings['UNKNOWN'].append({
                                    'term': term,
                                    'pattern': pattern,
                                    'episode': transcript_file.stem,
                                    'full_match': str(match)
                                })
                                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Summary
    print("\n\n=== SUMMARY OF FINDINGS ===")
    
    for proper_name, findings in sorted(all_findings.items()):
        if findings and proper_name != 'UNKNOWN':
            print(f"\n{proper_name}:")
            # Group by term
            term_counts = defaultdict(int)
            for f in findings:
                term_counts[f['term']] += f['count']
            
            for term, total_count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  '{term}' - {total_count} total occurrences")
    
    # Show unknown terms that might be worth investigating
    if 'UNKNOWN' in all_findings:
        print("\n\nPOTENTIAL NEW TERMS TO INVESTIGATE:")
        unknown_counts = Counter(f['term'] for f in all_findings['UNKNOWN'])
        for term, count in unknown_counts.most_common(20):
            if count >= 2:  # Appears in multiple contexts
                print(f"  '{term}' - {count} price contexts")
    
    # Save results
    output = {
        'findings': {k: v for k, v in all_findings.items()},
        'summary': {
            'total_episodes': len(transcript_files),
            'known_mappings_found': {k: len(v) for k, v in all_findings.items() if k != 'UNKNOWN'},
            'unknown_terms': len(all_findings.get('UNKNOWN', []))
        }
    }
    
    output_path = "data/episodes/bitcoin_dive_bar_analysis/quick_finance_scan.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")

if __name__ == "__main__":
    scan_transcripts("data/episodes/bitcoin_dive_bar_analysis/transcripts")