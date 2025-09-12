#!/usr/bin/env python3
"""
Run improved transcript analysis using modern models with large context windows
"""
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from research.improved_transcript_processor import ImprovedTranscriptProcessor


def load_transcripts():
    """Load all conference transcripts"""
    transcript_dir = Path("data/youtube_analysis/transcripts")
    transcripts = []
    
    for file_path in sorted(transcript_dir.glob("*_transcript.json")):
        if "claude" not in file_path.name:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check for processed version
            processed_path = file_path.parent / file_path.name.replace("_transcript.json", "_transcript_claude_postprocessed.json")
            if processed_path.exists():
                with open(processed_path, 'r') as f:
                    processed_data = json.load(f)
                    data['processed_text'] = processed_data.get('text', '')
            
            data['file_path'] = file_path.name
            transcripts.append(data)
    
    return transcripts


def main():
    print("=" * 70)
    print("IMPROVED TRANSCRIPT ANALYSIS - Using Modern LLMs")
    print("=" * 70)
    
    # Load transcripts
    print("\n1. Loading transcripts...")
    transcripts = load_transcripts()
    print(f"   Loaded {len(transcripts)} transcript files")
    
    total_chars = sum(len(t.get('processed_text', t.get('text', ''))) for t in transcripts)
    print(f"   Total content: {total_chars:,} characters (~{total_chars//4:,} tokens)")
    
    # Choose processing strategy
    print("\n2. Choosing model strategy...")
    print("   Available strategies:")
    print("   - deepseek-v3: 164K context, $0.20/M input (BEST VALUE)")
    print("   - mistral-small: 128K context, $0.25/M input")
    print("   - sonar-research: 128K context, $0.30/M input (research-optimized)")
    print("   - claude-sonnet: 200K context, $3.00/M input (expensive but powerful)")
    
    # Use Claude Sonnet for higher quality analysis
    strategy = "claude-sonnet"
    print(f"   Using: {strategy} (higher quality, ~$0.23 estimated)")
    
    processor = ImprovedTranscriptProcessor(strategy=strategy)
    
    # Analysis 1: Full context analysis for BitVM evolution
    print("\n3. Running full-context BitVM analysis...")
    
    bitvm_prompt = """
    Analyze these Bitcoin conference transcripts focusing on BitVM and Garbled Circuits.
    
    Provide a comprehensive technical analysis including:
    
    1. TECHNICAL EVOLUTION
       - How did BitVM evolve from v2 to v3?
       - What breakthrough enabled the 1000x improvement?
       - Specific performance metrics (before/after)
    
    2. GARBLED CIRCUITS IMPLEMENTATION
       - How do garbled circuits work in Bitcoin context?
       - Different implementation approaches (Yao, DelBraq, Glock, etc.)
       - Trade-offs between on-chain and off-chain costs
    
    3. KEY INNOVATIONS & BREAKTHROUGHS
       - Conditional disclosure of secrets
       - Verifiable encryption schemes
       - Circuit optimization techniques
    
    4. CHALLENGES & LIMITATIONS
       - Current bottlenecks
       - Security assumptions
       - Practical deployment issues
    
    5. FUTURE DIRECTIONS
       - Proposed improvements
       - Required Bitcoin upgrades (CTV, TXHASH, etc.)
       - Timeline expectations
    
    Output as detailed JSON with specific numbers, metrics, and technical details.
    """
    
    result1 = processor.analyze_with_full_context(transcripts, bitvm_prompt)
    
    # Save results
    output_dir = Path("research/outputs")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "deepseek_full_context_analysis.json", 'w') as f:
        json.dump(result1, f, indent=2)
    
    print(f"   Analyzed {result1.get('total_chars_analyzed', 0):,} characters")
    print(f"   Estimated cost: ${result1.get('cost', 0):.3f}")
    
    # Analysis 2: Iterative deep dive on specific topics
    print("\n4. Running iterative topic analysis...")
    
    focus_topics = [
        "garbled circuit",
        "snark verifier", 
        "bitvm bridge",
        "operator challenge",
        "groth16",
        "covenant",
        "taproot"
    ]
    
    result2 = processor.iterative_deep_analysis(transcripts, focus_topics)
    
    with open(output_dir / "deepseek_iterative_analysis.json", 'w') as f:
        json.dump(result2, f, indent=2)
    
    print(f"   Processed {result2['transcripts_processed']} transcripts")
    print(f"   Found insights on {len(result2.get('phase1_extracts', []))} topics")
    print(f"   Total cost: ${result2['total_cost']:.3f}")
    
    # Analysis 3: Find cross-references between talks
    print("\n5. Finding cross-references between talks...")
    
    references = processor.find_cross_references(transcripts)
    
    with open(output_dir / "cross_references.json", 'w') as f:
        json.dump(references, f, indent=2)
    
    if references:
        print(f"   Found references to: {', '.join(list(references.keys())[:5])}")
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total processing cost: ${processor.calculate_cost():.2f}")
    print(f"\nResults saved to research/outputs/:")
    print("  - deepseek_full_context_analysis.json")
    print("  - deepseek_iterative_analysis.json") 
    print("  - cross_references.json")
    
    # Cost comparison
    print(f"\nCost comparison:")
    print(f"  - This analysis (Claude Sonnet): ~${processor.calculate_cost():.2f}")
    print(f"  - With DeepSeek: ~${processor.calculate_cost() / 4:.2f}")
    print(f"  - With GPT-4: ~${processor.calculate_cost() * 1.5:.2f}")


if __name__ == "__main__":
    main()