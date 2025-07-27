#!/usr/bin/env python3
"""
Analyze all transcripts for financial terms
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.finance_term_discovery import FinanceTermDiscovery
import json

def main():
    # Create analyzer
    analyzer = FinanceTermDiscovery()
    
    # Transcript directory
    transcript_dir = "data/episodes/bitcoin_dive_bar_analysis/transcripts"
    
    print("Analyzing all transcripts for financial terms...")
    print("This may take several minutes...\n")
    
    # Analyze all transcripts
    all_results = analyzer.analyze_all_transcripts(transcript_dir)
    
    # Save comprehensive results
    output_path = "data/episodes/bitcoin_dive_bar_analysis/finance_terms_full_analysis.json"
    analyzer.save_results(all_results, output_path)
    
    # Show consolidated mappings
    print("\n=== CONSOLIDATED MAPPINGS ACROSS ALL EPISODES ===")
    with open(output_path, 'r') as f:
        full_results = json.load(f)
    
    # Group by actual ticker
    organized_mappings = {}
    for result in all_results:
        summary = result.get('summary', {})
        mappings = summary.get('potential_mappings', {})
        
        for ticker_desc, variants in mappings.items():
            # Extract likely ticker from description
            if 'MSTR' in ticker_desc and 'MicroStrategy' in ticker_desc:
                ticker = 'MSTR'
            elif 'MSTY' in ticker_desc:
                ticker = 'MSTY'
            elif 'SMLR' in ticker_desc or 'Semler' in ticker_desc:
                ticker = 'SMLR'
            elif 'Bitcoin' in ticker_desc or 'BTC' in ticker_desc:
                ticker = 'BTC'
            elif 'COIN' in ticker_desc:
                ticker = 'COIN'
            elif 'MARA' in ticker_desc:
                ticker = 'MARA'
            elif 'RIOT' in ticker_desc:
                ticker = 'RIOT'
            elif 'CLSK' in ticker_desc:
                ticker = 'CLSK'
            else:
                ticker = ticker_desc[:20] + '...' if len(ticker_desc) > 20 else ticker_desc
            
            if ticker not in organized_mappings:
                organized_mappings[ticker] = set()
            
            for variant in variants:
                if variant.get('confidence') in ['high', 'medium']:
                    organized_mappings[ticker].add(variant['term'])
    
    # Show organized results
    print("\n=== FINANCIAL TERM VARIANTS FOUND ===")
    for ticker in sorted(organized_mappings.keys()):
        terms = organized_mappings[ticker]
        if terms:
            print(f"\n{ticker}:")
            for term in sorted(terms):
                print(f"  - {term}")
    
    # Show statistics
    print("\n=== STATISTICS ===")
    total_terms = sum(len(terms) for terms in organized_mappings.values())
    print(f"Total unique variants found: {total_terms}")
    print(f"Tickers with variants: {len(organized_mappings)}")
    print(f"Episodes analyzed: {len(all_results)}")
    
    # Show most common unknown terms
    print("\n=== MOST FREQUENT POTENTIAL FINANCE TERMS ===")
    all_frequent = {}
    for result in all_results:
        frequent = result.get('frequent_terms', {})
        for term, count in frequent.items():
            if term not in all_frequent:
                all_frequent[term] = 0
            all_frequent[term] += count
    
    # Filter for potential finance terms
    finance_candidates = {}
    for term, count in all_frequent.items():
        # Look for patterns that suggest finance terms
        if (len(term) >= 3 and len(term) <= 8 and  # Ticker-like length
            count >= 20 and  # Appears frequently
            term not in ['like', 'right', 'know', 'think', 'just', 'that', 'this', 'will', 
                        'have', 'been', 'what', 'they', 'your', 'from', 'with', 'about',
                        'would', 'could', 'should', 'there', 'where', 'when', 'which']):
            finance_candidates[term] = count
    
    for term, count in sorted(finance_candidates.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {term}: {count} occurrences")

if __name__ == "__main__":
    main()