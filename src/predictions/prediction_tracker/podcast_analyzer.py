"""
Podcast Content Analyzer

Analyzes podcast transcripts to understand content and terminology,
then generates appropriate prompts for prediction extraction.
"""

import json
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re
from pathlib import Path

class PodcastAnalyzer:
    """Analyzes podcast content to generate appropriate extraction prompts"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    def analyze_podcast_series(self, transcript_files: List[str], sample_size: int = 3) -> Dict:
        """
        Analyze a podcast series to understand its content and terminology
        
        Args:
            transcript_files: List of transcript file paths
            sample_size: Number of episodes to sample for analysis
            
        Returns:
            Dictionary with podcast analysis including topics, terminology, patterns
        """
        # Sample episodes for analysis
        files_to_analyze = transcript_files[:sample_size] if len(transcript_files) > sample_size else transcript_files
        
        combined_analysis = {
            'topics': [],
            'asset_mentions': Counter(),
            'terminology': {},
            'prediction_patterns': [],
            'speaker_patterns': [],
            'podcast_type': '',
            'focus_areas': []
        }
        
        print(f"\n=== ANALYZING PODCAST SERIES ===")
        print(f"Sampling {len(files_to_analyze)} episodes for content analysis...")
        
        for file_path in files_to_analyze:
            print(f"\nAnalyzing: {Path(file_path).stem}")
            
            # Load transcript
            with open(file_path, 'r') as f:
                data = json.load(f)
                text = data.get('text', '')
            
            # Sample middle section (often most representative)
            text_length = len(text)
            sample_start = text_length // 3
            sample_end = sample_start + min(30000, text_length // 3)
            sample_text = text[sample_start:sample_end]
            
            # Analyze this episode
            episode_analysis = self._analyze_episode_content(sample_text)
            
            # Merge results
            combined_analysis['topics'].extend(episode_analysis.get('topics', []))
            combined_analysis['asset_mentions'].update(episode_analysis.get('assets', {}))
            combined_analysis['terminology'].update(episode_analysis.get('terminology', {}))
            combined_analysis['prediction_patterns'].extend(episode_analysis.get('patterns', []))
            combined_analysis['speaker_patterns'].extend(episode_analysis.get('speaker_patterns', []))
            combined_analysis['focus_areas'].extend(episode_analysis.get('focus_areas', []))
        
        # Determine podcast type based on analysis
        combined_analysis['podcast_type'] = self._determine_podcast_type(combined_analysis)
        
        # Clean up and deduplicate
        combined_analysis['topics'] = list(set(combined_analysis['topics']))
        combined_analysis['prediction_patterns'] = list(set(combined_analysis['prediction_patterns']))
        combined_analysis['speaker_patterns'] = list(set(combined_analysis['speaker_patterns']))
        combined_analysis['focus_areas'] = list(set(combined_analysis['focus_areas']))
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Podcast Type: {combined_analysis['podcast_type']}")
        print(f"Main Topics: {', '.join(combined_analysis['topics'][:5])}")
        print(f"Top Assets: {', '.join([f'{k}({v})' for k, v in combined_analysis['asset_mentions'].most_common(5)])}")
        
        return combined_analysis
    
    def _analyze_episode_content(self, text: str) -> Dict:
        """Use LLM to analyze episode content"""
        
        system_prompt = """Analyze this finance podcast transcript to understand:
1. Main topics discussed
2. Assets/tickers mentioned (and any slang/nicknames used)
3. How speakers make predictions (language patterns)
4. Terminology and jargon specific to this podcast
5. Overall focus (trading, investing, macro, crypto, etc)

Be thorough in identifying ALL variations of asset names and slang."""

        user_prompt = f"""Analyze this podcast transcript sample:

{text[:10000]}  # Limit to avoid token limits

Provide analysis in this JSON format:
{{
  "topics": ["list of main topics"],
  "assets": {{"TICKER": count, "SLANG_NAME": count}},
  "terminology": {{"slang_term": "actual_meaning"}},
  "patterns": ["how predictions are made"],
  "speaker_patterns": ["how speakers talk about prices"],
  "focus_areas": ["trading", "investing", "technical_analysis", etc]
}}"""

        # Get analysis from LLM
        response = self.llm_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"} if "gpt-4o" in self.llm_client.model else None
        )
        
        try:
            content = response['choices'][0]['message']['content']
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            # Fallback to regex-based analysis
            return self._fallback_analysis(text)
    
    def _fallback_analysis(self, text: str) -> Dict:
        """Basic regex-based analysis as fallback"""
        
        # Common finance/trading terms
        topics = []
        if re.search(r'bitcoin|btc|crypto|blockchain', text, re.I):
            topics.append('cryptocurrency')
        if re.search(r'stock|equity|share|market cap', text, re.I):
            topics.append('stocks')
        if re.search(r'option|call|put|strike', text, re.I):
            topics.append('options')
        if re.search(r'technical analysis|support|resistance|breakout', text, re.I):
            topics.append('technical_analysis')
            
        # Find potential tickers (uppercase 1-5 letter words)
        potential_tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        ticker_counts = Counter(potential_tickers)
        
        # Filter to likely tickers
        assets = {}
        common_words = {'I', 'A', 'THE', 'AND', 'OR', 'BUT', 'IF', 'TO', 'OF', 'IN', 'IT', 'IS', 'AT'}
        for ticker, count in ticker_counts.items():
            if ticker not in common_words and count > 2:
                assets[ticker] = count
        
        return {
            'topics': topics,
            'assets': dict(assets),
            'terminology': {},
            'patterns': ['price targets', 'timeframes'],
            'speaker_patterns': ['going to', 'will hit', 'target'],
            'focus_areas': topics
        }
    
    def _determine_podcast_type(self, analysis: Dict) -> str:
        """Determine podcast type based on analysis"""
        
        # Check asset mentions
        asset_mentions = analysis['asset_mentions']
        total_mentions = sum(asset_mentions.values())
        
        if total_mentions == 0:
            return 'general_finance'
        
        # Calculate percentages
        crypto_mentions = sum(v for k, v in asset_mentions.items() 
                            if k in ['BTC', 'ETH', 'BITCOIN', 'ETHEREUM', 'CRYPTO'])
        stock_mentions = sum(v for k, v in asset_mentions.items() 
                           if k not in ['BTC', 'ETH', 'BITCOIN', 'ETHEREUM', 'CRYPTO'] and len(k) <= 5)
        
        crypto_pct = crypto_mentions / total_mentions if total_mentions > 0 else 0
        
        # Determine type
        if crypto_pct > 0.7:
            return 'crypto_focused'
        elif crypto_pct > 0.3:
            return 'mixed_crypto_stocks'
        elif 'options' in analysis['topics'] or 'derivatives' in analysis['topics']:
            return 'options_trading'
        elif 'technical_analysis' in analysis['focus_areas']:
            return 'technical_trading'
        else:
            return 'general_investing'
    
    def generate_extraction_prompts(self, analysis: Dict) -> Tuple[str, str]:
        """
        Generate custom stage 1 and stage 2 prompts based on podcast analysis
        
        Returns:
            Tuple of (stage1_prompt, stage2_prompt)
        """
        podcast_type = analysis['podcast_type']
        assets = analysis['asset_mentions'].most_common(10)
        terminology = analysis['terminology']
        patterns = analysis['prediction_patterns']
        
        # Build asset mapping section
        asset_mapping = []
        for slang, actual in terminology.items():
            if actual.isupper() and len(actual) <= 10:  # Likely a ticker
                asset_mapping.append(f'- "{slang}" → {actual}')
        
        # Add common assets
        for asset, _ in assets:
            if asset not in terminology.values():
                asset_mapping.append(f'- "{asset.lower()}" → {asset}')
        
        # Generate Stage 1 prompt
        stage1_prompt = f"""Find locations in this {podcast_type.replace('_', ' ')} podcast where speakers mention price predictions.

CRITICAL: When in doubt, INCLUDE the position.
This podcast focuses on: {', '.join(analysis['topics'][:3])}

Common prediction patterns in this podcast:
{chr(10).join(f'- {p}' for p in patterns[:5])}

Look for mentions of these assets:
{chr(10).join(f'- {asset[0]}' for asset in assets[:10])}

Include ANY mention of:
- Price targets with these assets
- Future price movements
- Technical levels and targets
- Timeframe mentions (dates, periods, conditions)

Your job: Find ALL potential predictions, even borderline cases."""

        # Generate Stage 2 prompt
        stage2_prompt = f"""Extract price predictions from this {podcast_type.replace('_', ' ')} podcast snippet.

This podcast discusses: {', '.join(analysis['topics'][:3])}

ASSET MAPPING:
{chr(10).join(asset_mapping)}

PREDICTION PATTERNS TO LOOK FOR:
{chr(10).join(f'- {p}' for p in analysis['speaker_patterns'][:5])}

IMPORTANT CONTEXT:
- Focus areas: {', '.join(analysis['focus_areas'][:3])}
- Common terminology: {', '.join(f'{k} ({v})' for k, v in list(terminology.items())[:5])}

Extract predictions using the standard JSON schema.
Be inclusive with predictions - this podcast often uses casual language."""

        return stage1_prompt, stage2_prompt


class AdaptivePredictionExtractor:
    """Prediction extractor that adapts to different podcasts"""
    
    def __init__(self, snippet_model: str = "gpt-4o-mini", prediction_model: str = "gpt-4o"):
        from src.llm.llm_client import LLMClient
        
        self.snippet_client = LLMClient(model=snippet_model)
        self.prediction_client = LLMClient(model=prediction_model)
        self.analyzer = PodcastAnalyzer(self.snippet_client)
        
        # Cache for analyzed podcasts
        self.podcast_cache = {}
    
    def extract_predictions_from_podcast(self, transcript_files: List[str], 
                                       podcast_name: str = None) -> List[Dict]:
        """
        Extract predictions from a podcast series, adapting to its content
        
        Args:
            transcript_files: List of transcript JSON files
            podcast_name: Optional podcast name for caching
            
        Returns:
            List of predictions extracted
        """
        # Check cache
        if podcast_name and podcast_name in self.podcast_cache:
            print(f"Using cached analysis for {podcast_name}")
            analysis = self.podcast_cache[podcast_name]
        else:
            # Analyze the podcast
            analysis = self.analyzer.analyze_podcast_series(transcript_files)
            
            # Cache if name provided
            if podcast_name:
                self.podcast_cache[podcast_name] = analysis
        
        # Generate custom prompts
        stage1_prompt, stage2_prompt = self.analyzer.generate_extraction_prompts(analysis)
        
        print(f"\n=== GENERATED PROMPTS ===")
        print(f"\nStage 1 Preview:")
        print(stage1_prompt[:300] + "...")
        print(f"\nStage 2 Preview:") 
        print(stage2_prompt[:300] + "...")
        
        # Now use these prompts with the existing two-stage extractor
        # (We'll integrate this next)
        
        return {
            'analysis': analysis,
            'stage1_prompt': stage1_prompt,
            'stage2_prompt': stage2_prompt
        }