#!/usr/bin/env python3
"""
Step 3: Generate improved prompts based on ACTUAL Rogue Trader analysis
This reads the analysis results and creates new prompts
"""
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from llm.llm_client import OpenAIClient as LLMClient


def generate_prompts_from_analysis():
    """Generate improved prompts based on the actual analysis"""
    
    # Find the most recent analysis file
    analysis_dir = Path("data/episodes/rogue_trader_analysis")
    analysis_files = list(analysis_dir.glob("rogue_trader_analysis_*.json"))
    
    if not analysis_files:
        print("Error: No analysis files found. Run analyze_rogue_trader.py first!")
        return
    
    # Get the most recent analysis
    latest_analysis_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
    print(f"Using analysis from: {latest_analysis_file.name}")
    
    with open(latest_analysis_file, 'r') as f:
        analysis_data = json.load(f)
    
    print(f"Found analysis of {analysis_data['episodes_analyzed']} episodes")
    
    # Initialize LLM for prompt generation
    llm_client = LLMClient()
    llm_client.model = "gpt-4o"  # Using stronger model for synthesis
    
    # Combine all analyses
    all_analysis_text = []
    for episode in analysis_data['analyses']:
        for section in episode['sections_analyzed']:
            all_analysis_text.append(section['analysis'])
    
    combined_analysis = "\n\n".join(all_analysis_text)
    
    print("\nGenerating improved prompts based on Rogue Trader analysis...")
    
    # Generate prompts using the actual analysis
    system_prompt = """You are creating improved prompts for extracting price predictions from Bitcoin/crypto podcasts.
You have been given analysis of professional finance/trading podcasts.
Create prompts that incorporate the actual language patterns and terminology found."""

    user_prompt = f"""Based on this analysis of {analysis_data['episodes_analyzed']} Rogue Trader podcast episodes:

{combined_analysis[:15000]}  # Limit for token constraints

Create two prompts:

1. STAGE 1 PROMPT: For finding locations in transcripts where price predictions might be mentioned
2. STAGE 2 PROMPT: For extracting the actual predictions from those snippets

Make the prompts specific and incorporate the ACTUAL patterns found in the analysis.
Include real examples from the analysis.

Format as JSON:
{{
  "stage1_prompt": "...",
  "stage2_prompt": "...",
  "key_patterns_incorporated": ["list of specific patterns used"],
  "improvements_over_generic": ["list of improvements"]
}}"""

    response = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    try:
        prompts = json.loads(response['choices'][0]['message']['content'])
    except:
        print("Error parsing JSON response")
        prompts = {
            "stage1_prompt": "Error generating prompt",
            "stage2_prompt": "Error generating prompt"
        }
    
    # Add metadata
    prompts['metadata'] = {
        'generated_from': latest_analysis_file.name,
        'episodes_analyzed': analysis_data['episodes_analyzed'],
        'generation_date': datetime.now().isoformat(),
        'source_channel': 'Rogue Trader'
    }
    
    # Save the prompts
    prompts_file = analysis_dir / f"generated_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"\n{'='*80}")
    print("PROMPTS GENERATED")
    print('='*80)
    print(f"\nStage 1 Prompt Preview:")
    print(prompts['stage1_prompt'][:500] + "...")
    print(f"\nStage 2 Prompt Preview:")
    print(prompts['stage2_prompt'][:500] + "...")
    print(f"\nFull prompts saved to: {prompts_file}")
    
    # Also save a simple version for easy use
    simple_prompts_file = analysis_dir / "latest_prompts.json"
    with open(simple_prompts_file, 'w') as f:
        json.dump({
            'stage1': prompts['stage1_prompt'],
            'stage2': prompts['stage2_prompt']
        }, f, indent=2)
    
    print(f"Simple prompts saved to: {simple_prompts_file}")


if __name__ == "__main__":
    generate_prompts_from_analysis()