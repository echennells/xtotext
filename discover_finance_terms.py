#!/usr/bin/env python3
"""
Discover financial terms in Bitcoin Dive Bar transcripts
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
    
    # For testing, let's just analyze one transcript first
    print("Testing on Episode 5 first...")
    
    # Episode 5 where we know "similar" = SMLR
    test_transcript = "data/episodes/bitcoin_dive_bar_analysis/transcripts/Bitcoin Dive Bar EP 5 - Bitcoin 120k  Guests BTC SMLR MSTR_vuA8LrC_z3s_transcript.json"
    
    if Path(test_transcript).exists():
        result = analyzer.analyze_transcript(test_transcript)
        
        print("\n=== ANALYSIS RESULTS ===")
        print(f"Unknown terms found: {result['summary']['total_unknown_terms']}")
        print(f"High confidence terms: {result['summary']['high_confidence_terms']}")
        
        print("\n=== POTENTIAL MAPPINGS ===")
        for ticker, variants in result['summary']['potential_mappings'].items():
            print(f"\n{ticker}:")
            for variant in variants[:5]:  # Show first 5
                print(f"  - '{variant['term']}' ({variant['confidence']})")
                if variant.get('context'):
                    print(f"    Context: {variant['context'][:100]}...")
        
        print("\n=== TOP REPEATED UNKNOWNS ===")
        for term, count in list(result['summary']['top_repeated_unknowns'].items())[:10]:
            print(f"  {term}: {count} times")
        
        # Save results
        output_path = "data/episodes/bitcoin_dive_bar_analysis/finance_terms_analysis.json"
        analyzer.save_results([result], output_path)
        
        # Ask user before analyzing all
        response = input("\nAnalyze all transcripts? (y/n): ")
        
        if response.lower() == 'y':
            # Analyze all transcripts
            print("\n\nAnalyzing all transcripts...")
            all_results = analyzer.analyze_all_transcripts(transcript_dir)
            
            # Save comprehensive results
            output_path = "data/episodes/bitcoin_dive_bar_analysis/finance_terms_full_analysis.json"
            analyzer.save_results(all_results, output_path)
            
            # Show consolidated mappings
            print("\n=== CONSOLIDATED MAPPINGS ACROSS ALL EPISODES ===")
            with open(output_path, 'r') as f:
                full_results = json.load(f)
                
            for ticker, terms in full_results['discovered_mappings'].items():
                if terms:  # Only show if we found variants
                    print(f"\n{ticker}: {', '.join(terms)}")
        
    else:
        print(f"Test transcript not found: {test_transcript}")

if __name__ == "__main__":
    main()