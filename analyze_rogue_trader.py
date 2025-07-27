#!/usr/bin/env python3
"""
Step 2: Analyze Rogue Trader transcripts with stage1 model
This reads the ACTUAL transcripts and analyzes them
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from llm.llm_client import OpenAIClient as LLMClient


def analyze_transcripts():
    """Analyze the actual Rogue Trader transcripts"""
    
    # Find ALL transcripts in the directory
    output_dir = Path("data/episodes/rogue_trader_analysis")
    transcripts_dir = output_dir / "transcripts"
    
    if not transcripts_dir.exists():
        print("Error: No transcripts directory found. Run download_and_transcribe_rogue_trader.py first!")
        return
    
    # Get all transcript JSON files
    transcript_files = list(transcripts_dir.glob("*_transcript.json"))
    
    if not transcript_files:
        print("Error: No transcript files found in directory!")
        return
    
    print(f"Found {len(transcript_files)} transcripts to analyze")
    
    # Sort by name for consistent ordering
    transcript_files.sort(key=lambda x: x.name)
    
    # Initialize LLM client
    llm_client = LLMClient()
    llm_client.model = "gpt-4o-mini"  # Using stage1 model
    
    # Analyze each transcript
    all_analyses = []
    
    for i, transcript_file in enumerate(transcript_files):
        print(f"\n{'='*60}")
        print(f"Analyzing transcript {i+1}/{len(transcript_files)}")
        print(f"File: {transcript_file.name}")
        print('='*60)
        
        # Load transcript
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
            text = transcript_data.get('text', '')
        
        # Sample different parts of the transcript
        text_length = len(text)
        samples = {
            'beginning': text[:10000],
            'middle': text[text_length//2 - 5000:text_length//2 + 5000],
            'end': text[-10000:]
        }
        
        episode_analysis = {
            'file': transcript_file.name,
            'sections_analyzed': []
        }
        
        for section_name, section_text in samples.items():
            print(f"\nAnalyzing {section_name} section...")
            
            # Analyze with stage1 model
            system_prompt = """Analyze this finance/trading podcast transcript section. Extract:
1. Key financial terms and concepts discussed
2. How the speaker makes predictions or discusses future prices
3. Technical analysis terminology used
4. Time expressions (how they talk about timeframes)
5. Risk management language
6. Any unique slang or terminology

Be very specific and quote exact phrases when possible."""

            user_prompt = f"""Analyze this {section_name} section of a Rogue Trader podcast:

{section_text}

Provide a detailed analysis focusing on the language patterns used."""

            response = llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            analysis_text = response['choices'][0]['message']['content']
            
            # Save the raw analysis
            section_analysis = {
                'section': section_name,
                'analysis': analysis_text,
                'sample_length': len(section_text)
            }
            
            episode_analysis['sections_analyzed'].append(section_analysis)
            
            # Print summary
            print(f"Analysis excerpt: {analysis_text[:300]}...")
        
        all_analyses.append(episode_analysis)
    
    # Save all analyses
    analysis_output_file = output_dir / f"rogue_trader_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(analysis_output_file, 'w') as f:
        json.dump({
            'channel': 'Rogue Trader',
            'episodes_analyzed': len(all_analyses),
            'analysis_date': datetime.now().isoformat(),
            'analyses': all_analyses
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print('='*80)
    print(f"Analyzed {len(all_analyses)} episodes")
    print(f"Results saved to: {analysis_output_file}")
    print("\nNext step: Run generate_prompts_from_analysis.py to create improved prompts")
    
    return analysis_output_file


if __name__ == "__main__":
    analyze_transcripts()