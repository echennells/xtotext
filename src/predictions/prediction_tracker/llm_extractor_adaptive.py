"""
Adaptive Two-Stage LLM Prediction Extractor

Extends the two-stage extractor to automatically adapt to different podcast types
by analyzing content and generating custom prompts.
"""

import json
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .llm_extractor_two_stage import TwoStageLLMExtractor
from .podcast_analyzer import PodcastAnalyzer
from .models import Prediction, TimeFrame, Confidence, PredictionType


class AdaptiveTwoStageLLMExtractor(TwoStageLLMExtractor):
    """Two-stage extractor that adapts to podcast content"""
    
    def __init__(self, snippet_model: str = "gpt-4o-mini", prediction_model: str = "gpt-4o"):
        super().__init__(snippet_model, prediction_model)
        self.analyzer = PodcastAnalyzer(self.snippet_client)
        self.podcast_analysis_cache = {}
        self.custom_prompts_cache = {}
    
    def analyze_podcast_series(self, transcript_files: List[str], 
                             podcast_name: str = None, 
                             force_reanalyze: bool = False) -> Dict:
        """
        Analyze podcast series to understand content
        
        Args:
            transcript_files: List of transcript file paths
            podcast_name: Name for caching purposes
            force_reanalyze: Force re-analysis even if cached
            
        Returns:
            Analysis results
        """
        # Check cache
        if podcast_name and podcast_name in self.podcast_analysis_cache and not force_reanalyze:
            print(f"Using cached analysis for {podcast_name}")
            return self.podcast_analysis_cache[podcast_name]
        
        # Perform analysis
        analysis = self.analyzer.analyze_podcast_series(transcript_files)
        
        # Generate custom prompts
        stage1_prompt, stage2_prompt = self.analyzer.generate_extraction_prompts(analysis)
        
        # Store in cache
        if podcast_name:
            self.podcast_analysis_cache[podcast_name] = analysis
            self.custom_prompts_cache[podcast_name] = {
                'stage1': stage1_prompt,
                'stage2': stage2_prompt
            }
        
        return analysis
    
    def extract_predictions_from_file(self, transcript_file: str, episode_info: Dict,
                                    podcast_name: str = None,
                                    custom_prompts: Dict = None) -> List[Prediction]:
        """
        Extract predictions using adaptive prompts
        
        Args:
            transcript_file: Path to transcript JSON
            episode_info: Episode metadata
            podcast_name: Name of podcast for using cached prompts
            custom_prompts: Optional custom prompts to use
            
        Returns:
            List of extracted predictions
        """
        # Get custom prompts if not provided
        if not custom_prompts:
            if podcast_name and podcast_name in self.custom_prompts_cache:
                custom_prompts = self.custom_prompts_cache[podcast_name]
            else:
                # Use default prompts
                custom_prompts = None
        
        # Store prompts temporarily
        self._current_custom_prompts = custom_prompts
        
        # Call parent method
        return super().extract_predictions_from_file(transcript_file, episode_info)
    
    def find_prediction_locations(self, text: str, context: Dict) -> List[Dict]:
        """Override to use custom stage 1 prompt"""
        
        # Check for custom prompt
        if hasattr(self, '_current_custom_prompts') and self._current_custom_prompts:
            system_prompt = self._current_custom_prompts.get('stage1')
        else:
            # Use default prompt from parent
            system_prompt = """Find locations in this transcript where speakers mention ANY PRICE with future context.
CRITICAL: When in doubt, INCLUDE the position.
Better to have false positives than miss real predictions."""
        
        # Continue with parent logic but with custom prompt
        positions = []
        
        # Process in chunks if text is long
        MAX_CHUNK_SIZE = 50000
        if len(text) > MAX_CHUNK_SIZE:
            chunk_size = MAX_CHUNK_SIZE
            overlap = 2500
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                
                user_prompt = f"""Find potential price predictions in this transcript section:
{chunk}
Context: {json.dumps(context)}
CRITICAL: Extract VERY GENEROUS snippets - approximately 3 MINUTES of conversation (500-800 words) around each potential prediction."""
                
                # Track tokens
                user_tokens = self.count_tokens(user_prompt, self.snippet_client.model)
                self.token_usage['snippet_extraction']['input'] += user_tokens
                self.token_usage['snippet_extraction']['calls'] += 1
                
                # Make API call with custom system prompt
                chunk_snippets = self.snippet_client.extract_snippets(chunk, context, system_prompt)
                
                # Track output tokens
                output_tokens = self.count_tokens(str(chunk_snippets), self.snippet_client.model)
                self.token_usage['snippet_extraction']['output'] += output_tokens
                
                # Adjust positions for chunk offset
                for pos in chunk_snippets:
                    if isinstance(pos, dict) and 'position' in pos:
                        pos['position'] += i
                    positions.extend([pos] if isinstance(pos, dict) else [{'position': pos + i}])
                
                print(f"   Chunk {i//chunk_size + 1}: Found {len(chunk_snippets)} potential snippets")
                time.sleep(0.5)  # Rate limiting
        else:
            # Process entire text at once
            user_prompt = f"""Find potential price predictions in this transcript:
{text}
Context: {json.dumps(context)}
CRITICAL: Extract VERY GENEROUS snippets - approximately 3 MINUTES of conversation (500-800 words) around each potential prediction."""
            
            # Track tokens
            user_tokens = self.count_tokens(user_prompt, self.snippet_client.model)
            self.token_usage['snippet_extraction']['input'] += user_tokens
            self.token_usage['snippet_extraction']['calls'] += 1
            
            # Use custom system prompt
            snippets = self.snippet_client.extract_snippets(text, context, system_prompt)
            
            positions = snippets if isinstance(snippets, list) else []
        
        return positions
    
    def extract_predictions_from_snippets(self, snippets: List[Dict], episode_info: Dict) -> List[Prediction]:
        """Override to use custom stage 2 prompt"""
        
        # Get custom prompt if available
        custom_stage2_prompt = None
        if hasattr(self, '_current_custom_prompts') and self._current_custom_prompts:
            custom_stage2_prompt = self._current_custom_prompts.get('stage2')
        
        # If we have a custom prompt, we need to modify the prediction client behavior
        # For now, we'll use the parent method but could extend this
        # to pass the custom prompt to the prediction client
        
        # Temporarily store the custom prompt
        if custom_stage2_prompt:
            self.prediction_client._custom_system_prompt = custom_stage2_prompt
        
        try:
            # Call parent method
            predictions = super().extract_predictions_from_snippets(snippets, episode_info)
        finally:
            # Clean up
            if hasattr(self.prediction_client, '_custom_system_prompt'):
                delattr(self.prediction_client, '_custom_system_prompt')
        
        return predictions


def analyze_and_extract_podcast(podcast_name: str, 
                              transcript_files: List[str],
                              output_dir: str = "adaptive_predictions"):
    """
    Analyze a podcast and extract predictions adaptively
    
    Args:
        podcast_name: Name of the podcast
        transcript_files: List of transcript file paths
        output_dir: Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize extractor
    extractor = AdaptiveTwoStageLLMExtractor()
    
    print(f"\n{'='*60}")
    print(f"ADAPTIVE PREDICTION EXTRACTION: {podcast_name}")
    print(f"{'='*60}")
    
    # Step 1: Analyze the podcast
    print(f"\nStep 1: Analyzing podcast content...")
    analysis = extractor.analyze_podcast_series(
        transcript_files, 
        podcast_name=podcast_name
    )
    
    # Save analysis
    analysis_file = Path(output_dir) / f"{podcast_name}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")
    
    # Step 2: Extract predictions from each episode
    print(f"\nStep 2: Extracting predictions from {len(transcript_files)} episodes...")
    
    all_predictions = []
    
    for transcript_file in transcript_files:
        print(f"\n{'='*40}")
        print(f"Processing: {Path(transcript_file).stem}")
        
        # Load episode info
        with open(transcript_file, 'r') as f:
            data = json.load(f)
        
        episode_info = {
            'title': Path(transcript_file).stem,
            'date': data.get('upload_date', datetime.now().isoformat()),
            'video_id': data.get('id', '')
        }
        
        # Extract predictions
        predictions = extractor.extract_predictions_from_file(
            transcript_file,
            episode_info,
            podcast_name=podcast_name
        )
        
        print(f"Found {len(predictions)} predictions")
        
        # Convert to dict format
        for pred in predictions:
            pred_dict = pred.to_dict()
            pred_dict['podcast'] = podcast_name
            all_predictions.append(pred_dict)
    
    # Save all predictions
    predictions_file = Path(output_dir) / f"{podcast_name}_predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Results saved to: {predictions_file}")
    print(f"{'='*60}")
    
    # Print summary
    if all_predictions:
        # Asset distribution
        asset_counts = {}
        for pred in all_predictions:
            asset = pred.get('asset', 'Unknown')
            asset_counts[asset] = asset_counts.get(asset, 0) + 1
        
        print(f"\nAsset Distribution:")
        for asset, count in sorted(asset_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {asset}: {count}")
    
    return {
        'analysis': analysis,
        'predictions': all_predictions,
        'analysis_file': str(analysis_file),
        'predictions_file': str(predictions_file)
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_extractor_adaptive.py <podcast_name> [transcript_files...]")
        sys.exit(1)
    
    podcast_name = sys.argv[1]
    transcript_files = sys.argv[2:] if len(sys.argv) > 2 else []
    
    if not transcript_files:
        # Find transcript files in default location
        transcript_dir = Path("data/transcripts") / podcast_name
        if transcript_dir.exists():
            transcript_files = list(transcript_dir.glob("*.json"))
            transcript_files = [str(f) for f in transcript_files]
    
    if not transcript_files:
        print(f"No transcript files found for podcast: {podcast_name}")
        sys.exit(1)
    
    # Run extraction
    results = analyze_and_extract_podcast(podcast_name, transcript_files)
    print(f"\nDone! Check {results['predictions_file']} for results.")