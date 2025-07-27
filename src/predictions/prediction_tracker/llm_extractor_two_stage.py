"""
Two-stage LLM prediction extractor:
1. Cheap model (GPT-4.1-nano) finds potential prediction snippets
2. Reasoning model (o3-mini) extracts actual predictions from snippets
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import gc
import time
import tiktoken
import os
from pathlib import Path

from llm.llm_client import OpenAIClient
from .models import Prediction, PredictionType, Confidence, TimeFrame
from .sanity_checker import PredictionSanityChecker


class TwoStageLLMExtractor:
    """Two-stage prediction extractor with token usage tracking"""
    
    def __init__(self):
        # Create two clients with different models
        self.snippet_client = OpenAIClient()  # Will override to use GPT-4.1-nano
        self.prediction_client = OpenAIClient()  # Will use o3-mini
        
        # Override the models for cost optimization
        self.snippet_client.model = "gpt-4.1-nano"
        self.prediction_client.model = "o3-mini"
        
        # Token tracking
        self.token_usage = {
            'snippet_extraction': {'input': 0, 'output': 0, 'calls': 0},
            'prediction_extraction': {'input': 0, 'output': 0, 'calls': 0}
        }
        
        # Debug logging
        self.debug_log_dir = Path("logs/debug")
        self.debug_log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_file = None
        self.current_episode = None
        
        # Asset normalization mapping (updated with discovered slang)
        self.asset_map = {
            # Crypto
            'bitcoin': 'BTC', 'btc': 'BTC', 'big coin': 'BTC', 'bigcoin': 'BTC',
            'ethereum': 'ETH', 'eth': 'ETH', 'ether': 'ETH',
            'solana': 'SOL', 'sol': 'SOL',
            'chainlink': 'LINK', 'link': 'LINK',
            'cardano': 'ADA', 'ada': 'ADA',
            'ripple': 'XRP', 'xrp': 'XRP',
            'doge': 'DOGE', 'dogecoin': 'DOGE',
            'matic': 'MATIC', 'polygon': 'MATIC',
            
            # MSTR and related
            'mstr': 'MSTR', 'microstrategy': 'MSTR', 'micro strategy': 'MSTR', 'mst': 'MSTR',
            'msty': 'MSTY', 'misty': 'MSTY',  # The leveraged ETF
            
            # SMLR variants (from our scan)
            'smlr': 'SMLR', 'semler': 'SMLR', 'similar': 'SMLR', 'sampler': 'SMLR', 'summer': 'SMLR',
            'silver': 'SMLR',  # Common mistranscription when they say "similar"
            
            # Other Bitcoin companies
            'coin': 'COIN', 'coinbase': 'COIN',
            'mara': 'MARA', 'marathon': 'MARA',
            'riot': 'RIOT',
            'clsk': 'CLSK', 'cleanspark': 'CLSK',
            'hut': 'HUT', 'hut8': 'HUT',
            'bitf': 'BITF', 'bitfarms': 'BITF',
            'hive': 'HIVE',
            'gbtc': 'GBTC', 'grayscale': 'GBTC',
            'meta planet': 'METAPLANET', 'metaplanet': 'METAPLANET',
            
            # People (not assets, but good to recognize)
            'sailor': 'SAYLOR', 'saylor': 'SAYLOR',
        }
    
    def _init_debug_log(self, episode_title: str):
        """Initialize debug log for this episode"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_title = "".join(c for c in episode_title if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        self.debug_log_file = self.debug_log_dir / f"debug_{timestamp}_{safe_title}.json"
        self.current_episode = episode_title
        self.debug_data = {
            "episode": episode_title,
            "timestamp": timestamp,
            "models": {
                "snippet": self.snippet_client.model,
                "prediction": self.prediction_client.model
            },
            "snippets": [],
            "predictions": [],
            "errors": []
        }
    
    def _log_debug(self, stage: str, data: Dict):
        """Log debug data to file"""
        if self.debug_log_file and self.debug_data:
            if stage == "snippet":
                self.debug_data["snippets"].append(data)
            elif stage == "prediction":
                self.debug_data["predictions"].append(data)
            elif stage == "error":
                self.debug_data["errors"].append(data)
            
            # Write to file after each update
            with open(self.debug_log_file, 'w') as f:
                json.dump(self.debug_data, f, indent=2)
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count tokens for a given text and model"""
        try:
            # Use cl100k_base encoding for GPT-4 models
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # Fallback to approximate count
            return len(text) // 4
    
    def find_prediction_locations(self, text: str, context: Dict) -> List[Dict]:
        """
        Stage 1a: Find LOCATIONS where predictions might be
        Returns simple positions, no JSON needed
        """
        system_prompt = """Find locations in this transcript where speakers mention ANY PRICE with future context.

CRITICAL: When in doubt, INCLUDE the position.
Better to have false positives than miss real predictions.

Cast a wide net - look for ANY price mention with future context:
- "Bitcoin 120k" ✓ (include even without explicit timing)
- "SMLR to 50" ✓ (include casual predictions) 
- "I think we see 200" ✓ (include vague future references)
- "misty hits 100" ✓ (slang is fine)
- "similar going to 50" ✓ (mistranscriptions too)

Your job: Find ALL potential predictions, even borderline cases.
Stage 2 will do the precise filtering.

INCLUDE ANY mention of:
- Price numbers with assets (even without timing)
- "going to X", "hits X", "to X", "sees X"
- Future-sounding price mentions
- Slang terms: "misty" (MSTY), "similar/silver" (SMLR), "mst" (MSTR), "big coin" (BTC)

ONLY SKIP:
- Current prices ("it's at 115k now")
- Past prices ("was at 60k")

Return ONLY character positions as numbers, one per line.
Example output:
12450
45200
67890

Just the numbers, nothing else!"""

        all_positions = []
        
        # Process in 50k sections to keep responses small
        section_size = 50000
        overlap = 2500
        
        for start in range(0, len(text), section_size - overlap):
            end = min(start + section_size, len(text))
            section = text[start:end]
            
            if len(section.strip()) < 100:
                continue
                
            print(f"\n[LOCATION FINDING] Processing section {start:,}-{end:,} chars...")
            
            # Track token usage
            user_content = f"Find prediction locations in this transcript section:\n\n{section}"
            input_tokens = self.count_tokens(system_prompt + user_content, self.snippet_client.model)
            self.token_usage['snippet_extraction']['input'] += input_tokens
            self.token_usage['snippet_extraction']['calls'] += 1
            
            try:
                response = self.snippet_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                )
                
                content = response['choices'][0]['message']['content'].strip()
                
                # Track output tokens
                output_tokens = self.count_tokens(content, self.snippet_client.model)
                self.token_usage['snippet_extraction']['output'] += output_tokens
                
                # Parse simple number list
                for line in content.split('\n'):
                    line = line.strip()
                    if line and line.isdigit():
                        pos = int(line) + start  # Adjust for section offset
                        all_positions.append(pos)
                
                lines = content.split('\n')
                num_predictions = len([l for l in lines if l.strip().isdigit()])
                print(f"   Found {num_predictions} predictions in this section")
                
            except Exception as e:
                print(f"   Error in section {start}-{end}: {e}")
                continue
        
        # Deduplicate positions within 3000 chars (to avoid ANY overlapping snippets)
        # Since we extract 3000 chars around each position (1500 before, 1500 after),
        # positions need to be at least 3000 chars apart to have zero overlap
        unique_positions = []
        for pos in sorted(all_positions):
            if not unique_positions or pos - unique_positions[-1] >= 3000:
                unique_positions.append(pos)
        
        print(f"\n[LOCATION FINDING] Total: {len(unique_positions)} unique prediction locations")
        print(f"  (Deduplicated from {len(all_positions)} raw positions)")
        
        # Convert to expected format
        return [{"position": pos} for pos in unique_positions]
    
    def extract_snippet_around_position(self, text: str, position: int, context_chars: int = 3000) -> Dict:
        """
        Extract ~3 minutes of context around a position (about 500-800 words = ~3000 chars)
        """
        # We want context_chars total, with the prediction roughly in the middle
        # So extract context_chars/2 before and after the position
        half_context = context_chars // 2
        
        # Calculate start and end with bounds checking
        start = max(0, position - half_context)
        end = min(len(text), position + half_context)
        
        # If we hit the beginning, extend the end to maintain total context
        if start == 0:
            end = min(len(text), context_chars)
        # If we hit the end, extend the start
        elif end == len(text):
            start = max(0, len(text) - context_chars)
        
        # Find sentence boundaries to avoid cutting mid-sentence
        if start > 0:
            # Look for sentence start
            for i in range(start, min(start + 200, position)):
                if i > 0 and text[i-1:i+1] in ['. ', '? ', '! ', '\n\n']:
                    start = i + 1
                    break
        
        if end < len(text):
            # Look for sentence end
            for i in range(max(position, end - 200), end):
                if i < len(text) - 1 and text[i:i+2] in ['. ', '? ', '! ', '\n\n']:
                    end = i + 2
                    break
        
        return {
            'text': text[start:end],
            'char_start': start,
            'char_end': end
        }
    
    def extract_snippets(self, text: str, context: Dict, start_offset: int = 0) -> List[Dict]:
        """
        Stage 1: Use cheap model to find potential prediction snippets
        
        Returns list of snippets with their positions in the text
        """
        # Already using gpt-4.1-nano from __init__, no need to override
        
        system_prompt = """You are a snippet extractor for a Bitcoin/crypto podcast. 
Your job is to find parts of the conversation that MIGHT contain price predictions made BY THE SPEAKERS THEMSELVES.

IMPORTANT: Skip sections where they're quoting others ("Jane Street said", "Saylor thinks", "the guy said", "someone mentioned").

Look for sections where THE PODCAST SPEAKERS discuss:
- Future prices (going to, will hit, target, headed to, could reach, might see)
- Price levels (120k, 150k, $50, "two hundred", "triple digits", etc)
- Timeframes (next week, by end of year, in 3 months, this cycle, soon)
- Technical analysis (support, resistance, breakout, retracement, consolidation)
- Bullish/bearish sentiment with numbers or levels
- Asset mentions (BTC, Bitcoin, MSTR, "Misty", ETH, COIN, SMLR, Ethereum, MicroStrategy, etc.)
- Sequential predictions ("first this, then that", "after X then Y", "before we see")
- Conditional language ("if we break", "when volume picks up", "once we hit", "assuming")
- Market movements ("pump to", "dump to", "retrace to", "bounce off")
- Comparisons ("like last time when", "similar to 2021", "remember when it hit")

Extract snippets where SPEAKERS THEMSELVES (not quoting others) discuss:
- Specific price numbers (120k, $50, 200k - NOT vague like "100 plus")
- WITH future timeframes (next week, by end of year, this cycle, etc)
- For specific assets (BTC, Bitcoin, MSTR, SMLR, etc)

SKIP snippets that:
- Quote external sources ("Jane Street said", "according to X")
- Have NO timeframe context
- Use vague prices ("going higher", "100 plus", "triple digits")

IMPORTANT: Extract moderate context around predictions - approximately 30-45 SECONDS of conversation (roughly 150-200 words). 
This ensures the second model has enough context to understand:
- What asset is being discussed (could be mentioned earlier)
- The full context of the prediction
- Any conditions or reasoning

Include AT LEAST 3-5 sentences before and after the prediction mention to capture context.

Output SIMPLE JSON:
{
  "snippets": [
    {
      "text": "Moderate snippet: Include 30-45 seconds of conversation (150-200 words) with context before and after the prediction",
      "char_start": 0,
      "char_end": 1000,
      "reason": "mentions BTC going to 130k next week"
    }
  ]
}

CRITICAL: Keep total JSON response under 8000 characters to avoid truncation!

BE EXTREMELY GENEROUS - it's much better to extract too many snippets than miss predictions.
When in doubt, ALWAYS include it! False positives are fine, false negatives are absolutely not acceptable.
Even vague discussions about future prices should be included."""

        # Process in chunks if needed
        # GPT-4.1-nano has 1M token context, so we can use much larger chunks
        chunk_size = 400000 if "gpt-4.1-nano" in self.snippet_client.model else 100000
        snippets = []
        
        # Count input tokens
        system_tokens = self.count_tokens(system_prompt, self.snippet_client.model)
        
        # Remove the old chunking logic - we'll always use the full text
        # The extract_snippets_in_sections method handles large texts
        if False:  # Disabled old chunking
            print(f"\n[SNIPPET EXTRACTION] Processing {len(text):,} chars in chunks...")
            
            for i in range(0, len(text), chunk_size - 10000):  # 10k overlap
                chunk = text[i:i + chunk_size]
                
                user_prompt = f"""Find potential price predictions in this transcript section:

{chunk}

Context: {json.dumps(context)}

CRITICAL: Extract VERY GENEROUS snippets - approximately 3 MINUTES of conversation (500-800 words) around each potential prediction.
Include AT LEAST 10-15 sentences before and after any price mention to ensure full context about what asset is being discussed."""
                
                # Track tokens
                user_tokens = self.count_tokens(user_prompt, self.snippet_client.model)
                self.token_usage['snippet_extraction']['input'] += system_tokens + user_tokens
                self.token_usage['snippet_extraction']['calls'] += 1
                
                # Make API call with custom system prompt
                chunk_snippets = self.snippet_client.extract_snippets(chunk, context, system_prompt)
                
                # Track output tokens (approximate)
                output_tokens = self.count_tokens(str(chunk_snippets), self.snippet_client.model)
                self.token_usage['snippet_extraction']['output'] += output_tokens
                
                snippets.extend(chunk_snippets)
                
                print(f"   Chunk {i//chunk_size + 1}: Found {len(chunk_snippets)} potential snippets")
                time.sleep(0.5)  # Rate limiting
        else:
            user_prompt = f"""Find potential price predictions in this transcript:

{text}

Context: {json.dumps(context)}

CRITICAL: Extract VERY GENEROUS snippets - approximately 3 MINUTES of conversation (500-800 words) around each potential prediction.
Include AT LEAST 10-15 sentences before and after any price mention to ensure full context about what asset is being discussed."""
            
            # Track tokens
            user_tokens = self.count_tokens(user_prompt, self.snippet_client.model)
            self.token_usage['snippet_extraction']['input'] += system_tokens + user_tokens
            self.token_usage['snippet_extraction']['calls'] += 1
            
            # Use the new extract_snippets method
            snippets = self.snippet_client.extract_snippets(text, context, system_prompt)
            
            # Adjust character positions if we have a start_offset
            if start_offset > 0:
                for snippet in snippets:
                    if 'char_start' in snippet:
                        snippet['char_start'] += start_offset
                    if 'char_end' in snippet:
                        snippet['char_end'] += start_offset
            
            # Track output tokens
            output_tokens = self.count_tokens(str(snippets), self.snippet_client.model)
            self.token_usage['snippet_extraction']['output'] += output_tokens
        
        print(f"\n[SNIPPET EXTRACTION] Found {len(snippets)} potential prediction snippets")
        print(f"[TOKEN USAGE] Snippet extraction - Input: {self.token_usage['snippet_extraction']['input']:,}, Output: {self.token_usage['snippet_extraction']['output']:,}")
        
        # Log debug data for all snippets
        for i, snippet in enumerate(snippets):
            self._log_debug("snippet", {
                "index": i,
                "text": snippet.get('text', ''),
                "reason": snippet.get('reason', ''),
                "char_start": snippet.get('char_start', 0),
                "char_end": snippet.get('char_end', 0),
                "tokens": self.count_tokens(snippet.get('text', ''), self.snippet_client.model)
            })
        
        return snippets
    
    def extract_predictions_from_snippets(self, snippets: List[Dict], episode_info: Dict) -> List[Prediction]:
        """
        Stage 2: Use expensive model to extract actual predictions from snippets
        """
        all_raw_predictions = []
        
        print(f"\n[PREDICTION EXTRACTION] Processing {len(snippets)} snippets with expensive model...")
        
        for i, snippet in enumerate(snippets):
            snippet_text = snippet.get('text', '')
            if not snippet_text:
                continue
            
            print(f"\n   Snippet {i+1}/{len(snippets)} ({len(snippet_text)} chars)")
            print(f"   Reason: {snippet.get('reason', 'No reason given')}")
            print(f"\n   --- FULL SNIPPET TEXT ---")
            print(snippet_text)
            print(f"   --- END SNIPPET ---")
            
            # Count tokens for this snippet  
            # Get system prompt from the client's extract_predictions method
            system_tokens = 500  # Approximate for now since we can't easily access the internal prompt
            user_tokens = self.count_tokens(snippet_text, self.prediction_client.model)
            self.token_usage['prediction_extraction']['input'] += system_tokens + user_tokens
            self.token_usage['prediction_extraction']['calls'] += 1
            
            # Extract predictions using the expensive model
            context = {
                'episode_title': episode_info.get('title', ''),
                'episode_date': episode_info.get('date', ''),
                'snippet_position': snippet.get('position', 'unknown'),
                'snippet_timestamp_start': snippet.get('timestamp_start'),
                'snippet_timestamp_end': snippet.get('timestamp_end')
            }
            
            predictions = self.prediction_client.extract_predictions(snippet_text, context)
            
            # Track output tokens
            output_tokens = self.count_tokens(str(predictions), self.prediction_client.model)
            self.token_usage['prediction_extraction']['output'] += output_tokens
            
            print(f"   Found {len(predictions)} predictions in this snippet")
            
            # Log debug data for this snippet's predictions
            self._log_debug("prediction", {
                "snippet_index": i,
                "snippet_text": snippet_text,
                "snippet_reason": snippet.get('reason', ''),
                "predictions_found": len(predictions),
                "raw_predictions": predictions,
                "tokens_used": {
                    "input": user_tokens,
                    "output": output_tokens
                }
            })
            
            # Add snippet timing info to each prediction
            for pred in predictions:
                pred['snippet_timestamp_start'] = snippet.get('timestamp_start')
                pred['snippet_timestamp_end'] = snippet.get('timestamp_end')
            
            # Just collect all predictions without deduplication
            all_raw_predictions.extend(predictions)
            
            # Rate limiting
            time.sleep(0.5)
        
        print(f"\n[PREDICTION EXTRACTION] Total raw predictions: {len(all_raw_predictions)}")
        
        # Now deduplicate and convert to prediction objects
        final_predictions = self._deduplicate_and_convert_predictions(all_raw_predictions, episode_info)
        
        print(f"[PREDICTION EXTRACTION] After deduplication: {len(final_predictions)} unique predictions")
        print(f"[TOKEN USAGE] Prediction extraction - Input: {self.token_usage['prediction_extraction']['input']:,}, Output: {self.token_usage['prediction_extraction']['output']:,}")
        
        return final_predictions
    
    def _deduplicate_and_convert_predictions(self, raw_predictions: List[Dict], episode_info: Dict) -> List[Prediction]:
        """Deduplicate raw predictions and convert to Prediction objects"""
        seen_predictions = set()
        final_predictions = []
        
        print(f"\n[DEDUPLICATION] Processing {len(raw_predictions)} raw predictions...")
        
        for pred in raw_predictions:
            # Create a comprehensive key based on the raw prediction data
            # Include asset, price, and a normalized quote to catch duplicates
            quote = pred.get('quote', '').strip()[:100]  # First 100 chars of quote
            asset = pred.get('asset', '').upper()
            price = pred.get('price', 0)
            
            # Comprehensive key - asset, price, and quote
            # This prevents missing different price predictions with similar wording
            key = f"{asset}|{price}|{quote}"
            
            if key not in seen_predictions:
                seen_predictions.add(key)
                
                # Try to create prediction object
                prediction_obj = self._create_prediction_object(pred, episode_info)
                if prediction_obj:
                    # Validate that we have a quote - essential for verification and timestamp matching
                    if not prediction_obj.raw_text or len(prediction_obj.raw_text.strip()) < 10:
                        print(f"   ✗ Rejected: {prediction_obj.asset} ${prediction_obj.value:,.0f} - No quote provided")
                        self._log_debug("error", {
                            "stage": "prediction_validation", 
                            "reason": "No quote provided by LLM",
                            "asset": prediction_obj.asset,
                            "value": prediction_obj.value,
                            "prediction": pred
                        })
                        continue
                    
                    final_predictions.append(prediction_obj)
                    try:
                        # Format time_frame safely (it's an enum)
                        time_frame_str = prediction_obj.time_frame.value if prediction_obj.time_frame else "No timeframe"
                        print(f"   ✓ {prediction_obj.asset} ${prediction_obj.value:,.0f} - {time_frame_str}")
                    except Exception as e:
                        print(f"   ✓ {prediction_obj.asset} ${prediction_obj.value} - {prediction_obj.time_frame} (formatting error: {e})")
                else:
                    print(f"   ✗ Failed to convert: {asset} {pred.get('price', 'unknown price')}")
                    self._log_debug("error", {
                        "stage": "prediction_conversion",
                        "raw_prediction": pred,
                        "reason": "Failed to create prediction object"
                    })
            else:
                print(f"   - Duplicate skipped: {asset} (same quote)")
        
        return final_predictions
    
    def extract_predictions_from_file(self, transcript_file: str, episode_info: Dict) -> List[Prediction]:
        """
        Main entry point: Extract predictions using two-stage approach
        """
        print(f"\n=== TWO-STAGE PREDICTION EXTRACTION ===")
        print(f"Episode: {episode_info.get('title', 'Unknown')}")
        
        # Initialize debug logging for this episode
        self._init_debug_log(episode_info.get('title', 'Unknown'))
        
        # Reset token usage for this episode
        self.token_usage = {
            'snippet_extraction': {'input': 0, 'output': 0, 'calls': 0},
            'prediction_extraction': {'input': 0, 'output': 0, 'calls': 0}
        }
        
        # Load transcript text
        with open(transcript_file, 'r') as f:
            data = json.load(f)
            text = data.get('text', '')
            segments = data.get('segments', [])
        
        print(f"Transcript length: {len(text):,} characters")
        
        # Stage 1: Find prediction locations, then extract 3-minute snippets
        print("\n[STAGE 1] Finding prediction locations...")
        locations = self.find_prediction_locations(text, episode_info)
        
        if not locations:
            print("No potential predictions found")
            return []
        
        # Extract 3 minutes of context around each location
        print(f"\n[STAGE 1] Extracting 3-minute context around {len(locations)} locations...")
        
        # Build character position to segment mapping
        char_to_segment = {}
        if segments:
            char_pos = 0
            for seg_idx, segment in enumerate(segments):
                seg_text = segment.get('text', '')
                seg_start = segment.get('start', 0)
                seg_end = segment.get('end', 0)
                seg_len = len(seg_text)
                for i in range(seg_len):
                    char_to_segment[char_pos + i] = {
                        'segment_idx': seg_idx,
                        'start_time': seg_start,
                        'end_time': seg_end
                    }
                char_pos += seg_len + 1  # +1 for space between segments
        
        snippets = []
        for i, loc in enumerate(locations):
            position = loc.get('position', 0)
            snippet = self.extract_snippet_around_position(text, position)
            
            # Find the timestamp for this snippet based on character position
            snippet_center = (snippet['char_start'] + snippet['char_end']) // 2
            if snippet_center in char_to_segment:
                seg_info = char_to_segment[snippet_center]
                snippet['timestamp_start'] = seg_info['start_time']
                snippet['timestamp_end'] = seg_info['end_time']
                snippet['segment_idx'] = seg_info['segment_idx']
            else:
                # Fallback: find nearest character position
                nearest_pos = min(char_to_segment.keys(), key=lambda x: abs(x - snippet_center)) if char_to_segment else None
                if nearest_pos and abs(nearest_pos - snippet_center) < 500:  # Within 500 chars
                    seg_info = char_to_segment[nearest_pos]
                    snippet['timestamp_start'] = seg_info['start_time']
                    snippet['timestamp_end'] = seg_info['end_time']
                    snippet['segment_idx'] = seg_info['segment_idx']
            
            # Extract a preview from the snippet text for debugging
            preview_start = max(0, position - snippet['char_start'])
            preview_end = min(preview_start + 100, len(snippet['text']))
            snippet['reason'] = snippet['text'][preview_start:preview_end].strip()
            snippets.append(snippet)
            print(f"   Location {i+1}: position {position:,} → snippet {snippet['char_start']:,}-{snippet['char_end']:,} ({len(snippet['text'])} chars)")
        
        if not snippets:
            print("No potential prediction snippets found")
            return []
        
        # Stage 2: Extract predictions from snippets with expensive model
        predictions = self.extract_predictions_from_snippets(snippets, episode_info)
        
        # Timestamp matching is now handled during prediction creation using snippet timestamps
        print(f"\n[TIMESTAMP INFO] All predictions have timestamps from their source snippets")
        
        # Print final token usage summary
        print(f"\n=== TOKEN USAGE SUMMARY ===")
        print(f"Snippet Extraction ({self.snippet_client.model}):")
        print(f"  Calls: {self.token_usage['snippet_extraction']['calls']}")
        print(f"  Input tokens: {self.token_usage['snippet_extraction']['input']:,}")
        print(f"  Output tokens: {self.token_usage['snippet_extraction']['output']:,}")
        print(f"  Total tokens: {self.token_usage['snippet_extraction']['input'] + self.token_usage['snippet_extraction']['output']:,}")
        
        print(f"\nPrediction Extraction ({self.prediction_client.model}):")
        print(f"  Calls: {self.token_usage['prediction_extraction']['calls']}")
        print(f"  Input tokens: {self.token_usage['prediction_extraction']['input']:,}")
        print(f"  Output tokens: {self.token_usage['prediction_extraction']['output']:,}")
        print(f"  Total tokens: {self.token_usage['prediction_extraction']['input'] + self.token_usage['prediction_extraction']['output']:,}")
        
        # Cost estimation
        # GPT-4.1-nano: $0.10/1M input, $0.40/1M output
        nano_cost = (self.token_usage['snippet_extraction']['input'] / 1_000_000) * 0.10 + \
                   (self.token_usage['snippet_extraction']['output'] / 1_000_000) * 0.40
        
        # gpt-4o: $2.50/1M input, $10/1M output
        gpt4_cost = (self.token_usage['prediction_extraction']['input'] / 1_000_000) * 2.50 + \
                    (self.token_usage['prediction_extraction']['output'] / 1_000_000) * 10.0
        
        print(f"\nEstimated costs:")
        print(f"  GPT-4.1-nano: ${nano_cost:.4f}")
        print(f"  {self.prediction_client.model}: ${gpt4_cost:.4f}")
        print(f"  Total: ${nano_cost + gpt4_cost:.4f}")
        
        # Final debug log summary
        if self.debug_data:
            self.debug_data["summary"] = {
                "total_snippets": len(self.debug_data["snippets"]),
                "total_predictions": len(predictions),
                "token_usage": self.token_usage,
                "cost": {
                    "snippet_extraction": nano_cost,
                    "prediction_extraction": gpt4_cost,
                    "total": nano_cost + gpt4_cost
                }
            }
            self._log_debug("summary", {})
            print(f"\n[DEBUG LOG] Saved to: {self.debug_log_file}")
        
        # Sanity check predictions before returning
        if predictions:
            sanity_checker = PredictionSanityChecker(debug_logger=self._log_debug if self.debug_data else None)
            predictions = sanity_checker.check_predictions(predictions, episode_info)
        
        return predictions
    
    def _find_quote_in_segments(self, quote: str, segments: List[Dict]) -> Optional[Dict]:
        """Find which segment(s) contain a quote and return timing info"""
        quote_lower = quote.lower().strip()
        quote_words = quote_lower.split()
        
        if not quote_words:
            return None
        
        best_match = None
        best_score = 0
        
        for i, segment in enumerate(segments):
            segment_text = segment.get('text', '').lower().strip()
            
            matching_words = sum(1 for word in quote_words if word in segment_text)
            score = matching_words / len(quote_words) if quote_words else 0
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = {
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'segment_index': i
                }
                
                if score < 0.9 and i < len(segments) - 1:
                    next_segment_text = segments[i + 1].get('text', '').lower()
                    combined_text = segment_text + " " + next_segment_text
                    combined_score = sum(1 for word in quote_words if word in combined_text) / len(quote_words)
                    
                    if combined_score > 0.8:
                        best_match['end'] = segments[i + 1].get('end', best_match['end'])
        
        return best_match
    
    def _create_prediction_object(self, pred_data: Dict, episode_info: Dict) -> Optional[Prediction]:
        """Convert LLM output to Prediction object"""
        try:
            asset_raw = pred_data.get('asset', '').lower()
            asset = self.asset_map.get(asset_raw, asset_raw.upper())
            
            if not asset or len(asset) > 10:
                print(f"   [DEBUG] Asset '{asset}' rejected (empty or too long)")
                return None
            
            # LLM should return clean numeric values
            try:
                price = float(pred_data.get('price', 0))
                print(f"   [DEBUG] Price from LLM: {price}")
            except (ValueError, TypeError) as e:
                print(f"   [DEBUG] Price parsing error: {e}")
                price = 0
            
            if price <= 0:
                print(f"   [DEBUG] Price {price} rejected (<= 0)")
                return None
            
            # Fix common price interpretation issues for Bitcoin
            if asset == 'BTC':
                # Handle decimal notation (1.4, 1.7, etc. meaning 140k, 170k)
                # If price is between 1 and 10, it's likely decimal notation for hundreds of thousands
                if 1 <= price <= 10:
                    price = price * 100000
                    print(f"   [DEBUG] Normalized decimal notation: {price/100000:.1f} -> ${price:,.0f}")
                
                # Handle small numbers that are likely in thousands (e.g., "120" means 120k)
                elif price < 1000:
                    price = price * 1000
                    print(f"   [DEBUG] Normalized to thousands: {price/1000:.0f} -> ${price:,.0f}")
                
                # Handle numbers like 1700 that might be 170k (based on context like "1.70")
                elif 1000 <= price < 10000:
                    # Check if this looks like it came from decimal notation
                    # Numbers like 1400, 1700 likely came from "1.4", "1.7"
                    if price % 100 == 0 and 1000 <= price <= 3000:
                        price = price * 100
                        print(f"   [DEBUG] Normalized likely decimal: {price/100000:.1f} -> ${price:,.0f}")
                
                # Special case for teen numbers that might be 110k-190k
                elif 10 <= price <= 20:
                    price = (price + 100) * 1000
                    print(f"   [DEBUG] Normalized teen number: {(price/1000-100):.0f} -> ${price:,.0f}")
            
            # Parse timeframe from structured data
            timeframe_str = pred_data.get('timeframe', '').strip()
            timeframe_type = pred_data.get('timeframe_type', '').strip()
            timeframe_value = pred_data.get('timeframe_value')
            
            # REJECT predictions without timeframes
            if not timeframe_str or timeframe_str.lower() in ['none', 'null', '', 'no timeframe']:
                print(f"   [DEBUG] Rejecting prediction - no timeframe: {asset} ${price}")
                return None
            
            # Use structured timeframe data if available, otherwise fall back to parsing
            if timeframe_type:
                timeframe_data = self._parse_structured_timeframe(
                    timeframe_str, timeframe_type, timeframe_value,
                    episode_info.get('date', datetime.now().isoformat())
                )
            else:
                # Fallback to old parser for backwards compatibility
                timeframe_data = self._parse_timeframe(
                    timeframe_str,
                    episode_info.get('date', datetime.now().isoformat())
                )
            
            # Parse confidence
            confidence_str = pred_data.get('confidence', 'uncertain').lower()
            confidence_map = {
                'high': Confidence.HIGH,
                'medium': Confidence.MEDIUM,
                'low': Confidence.LOW,
                'uncertain': Confidence.UNCERTAIN
            }
            confidence = confidence_map.get(confidence_str, Confidence.UNCERTAIN)
            
            # Ensure price is float before creating prediction
            if not isinstance(price, (int, float)):
                print(f"   [DEBUG] Price type error: got {type(price)} instead of float")
                return None
                
            # Use snippet timestamp if available
            timestamp_start = pred_data.get('snippet_timestamp_start')
            if timestamp_start is not None:
                # Format the timestamp
                formatted_timestamp = self._format_timestamp(timestamp_start)
            else:
                # Fallback to LLM-provided timestamp
                formatted_timestamp = pred_data.get('timestamp', 'unknown')
            
            # Create prediction
            prediction = Prediction(
                asset=asset,
                prediction_type=PredictionType.PRICE_TARGET,
                value=float(price),  # Explicitly cast to float
                predicted_date=timeframe_data.get('predicted_date'),
                time_frame=timeframe_data.get('time_frame'),
                time_value=timeframe_data.get('time_value'),
                confidence=confidence,
                reasoning=pred_data.get('reasoning'),
                episode=episode_info.get('title', ''),
                episode_date=episode_info.get('date', ''),
                timestamp=formatted_timestamp,
                timestamp_start=timestamp_start,
                timestamp_end=pred_data.get('snippet_timestamp_end'),
                raw_text=pred_data.get('quote', '')[:500],
                prediction_id=self._generate_id(asset, price, timeframe_data.get('predicted_date'))
            )
            
            # Generate YouTube link if we have video ID and timestamp
            video_id = episode_info.get('video_id')
            if video_id and timestamp_start is not None:
                prediction.youtube_link = prediction.generate_youtube_link(video_id)
            
            return prediction
            
        except (ValueError, KeyError, TypeError) as e:
            return None
    
    def _parse_structured_timeframe(self, timeframe_str: str, timeframe_type: str, 
                                     timeframe_value: any, episode_date: str) -> Dict:
        """Parse structured timeframe data from LLM"""
        result = {}
        
        try:
            current_date = datetime.fromisoformat(episode_date.split('T')[0])
        except:
            current_date = datetime.now()
            
        if timeframe_type == 'specific_date':
            if isinstance(timeframe_value, str) and '-' in timeframe_value:
                # Already in YYYY-MM-DD format
                result['predicted_date'] = timeframe_value
                result['time_frame'] = TimeFrame.SPECIFIC_DATE
            else:
                # Need to parse "July 18th" style - use the original parser
                return self._parse_timeframe(timeframe_str, episode_date)
                
        elif timeframe_type == 'days':
            result['time_frame'] = TimeFrame.DAYS
            result['time_value'] = int(timeframe_value) if timeframe_value else 7
            
        elif timeframe_type == 'weeks':
            result['time_frame'] = TimeFrame.WEEKS
            result['time_value'] = int(timeframe_value) if timeframe_value else 2
            
        elif timeframe_type == 'months':
            result['time_frame'] = TimeFrame.MONTHS
            result['time_value'] = int(timeframe_value) if timeframe_value else 3
            
        elif timeframe_type == 'conditional':
            # For conditional predictions, use a reasonable default
            result['time_frame'] = TimeFrame.MONTHS
            result['time_value'] = 2
            
        elif timeframe_type == 'market_cycle':
            # Bull runs typically last 6-12 months
            result['time_frame'] = TimeFrame.MONTHS
            result['time_value'] = 6
            
        else:
            # Unknown type, fallback to parser
            return self._parse_timeframe(timeframe_str, episode_date)
            
        return result
    
    def _parse_timeframe(self, timeframe_str: str, episode_date: str) -> Dict:
        """Parse timeframe string into structured data"""
        timeframe_lower = timeframe_str.lower()
        result = {}
        
        # Try to parse specific dates
        if any(month in timeframe_lower for month in [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]):
            for month_num, month_name in enumerate([
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december'
            ], 1):
                if month_name in timeframe_lower:
                    import re
                    day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?', timeframe_lower)
                    day = int(day_match.group(1)) if day_match else 1
                    
                    try:
                        current_date = datetime.fromisoformat(episode_date)
                    except:
                        current_date = datetime.now()
                    
                    year = current_date.year
                    if month_num < current_date.month:
                        year += 1
                    
                    predicted_date = f"{year}-{month_num:02d}-{day:02d}"
                    result['predicted_date'] = predicted_date
                    result['time_frame'] = TimeFrame.SPECIFIC_DATE
                    break
        
        # Relative timeframes
        elif 'tomorrow' in timeframe_lower:
            try:
                current_date = datetime.fromisoformat(episode_date)
            except:
                current_date = datetime.now()
            next_day = current_date.replace(day=current_date.day + 1)
            result['predicted_date'] = next_day.isoformat()[:10]
            result['time_frame'] = TimeFrame.SPECIFIC_DATE
        
        elif 'next week' in timeframe_lower or 'by next week' in timeframe_lower:
            result['time_frame'] = TimeFrame.WEEKS
            result['time_value'] = 1
        
        elif 'end of year' in timeframe_lower or 'eoy' in timeframe_lower or 'this year' in timeframe_lower or 'by year' in timeframe_lower:
            result['time_frame'] = TimeFrame.EOY
            try:
                current_date = datetime.fromisoformat(episode_date)
            except:
                current_date = datetime.now()
            result['predicted_date'] = f"{current_date.year}-12-31"
        
        elif 'month' in timeframe_lower:
            import re
            num_match = re.search(r'(\d+)\s*month', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = int(num_match.group(1))
            elif 'next month' in timeframe_lower:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = 1
        
        elif 'week' in timeframe_lower:
            import re
            num_match = re.search(r'(\d+)\s*week', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = int(num_match.group(1))
            elif 'this week' in timeframe_lower:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = 0
        
        elif 'day' in timeframe_lower:
            import re
            num_match = re.search(r'(\d+)\s*day', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = int(num_match.group(1))
        
        # Handle specific contextual timeframes
        elif 'imminently' in timeframe_lower or 'any day' in timeframe_lower:
            result['time_frame'] = TimeFrame.DAYS
            result['time_value'] = 3  # Very near term
            
        elif 'very soon' in timeframe_lower or 'right now' in timeframe_lower:
            result['time_frame'] = TimeFrame.DAYS
            result['time_value'] = 7  # Within a week
            
        elif 'near-term' in timeframe_lower or 'short term' in timeframe_lower:
            result['time_frame'] = TimeFrame.WEEKS
            result['time_value'] = 2  # Couple weeks
            
        elif 'bull run' in timeframe_lower or 'this cycle' in timeframe_lower:
            result['time_frame'] = TimeFrame.MONTHS
            result['time_value'] = 6  # Bull cycles typically last months
            
        elif 'after' in timeframe_lower or 'once' in timeframe_lower or 'when' in timeframe_lower:
            # Conditional - more varied based on context
            if 'tomorrow' in timeframe_lower:
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = 1
            elif 'week' in timeframe_lower:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = 1
            else:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = 2  # Conditional events typically happen in months
            
        # Handle market momentum context
        elif 'momentum' in timeframe_lower or 'current' in timeframe_lower:
            result['time_frame'] = TimeFrame.WEEKS
            result['time_value'] = 2  # Current momentum plays out in weeks
        
        # If no specific timeframe matched but there's text, vary by context
        elif timeframe_str:
            # Don't always default to same values
            text_len = len(timeframe_str)
            if text_len < 10:  # Very short like "soon"
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = 14
            elif text_len < 25:  # Medium phrases
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = 3
            else:  # Longer conditional phrases
                result['time_frame'] = TimeFrame.MONTHS  
                result['time_value'] = 1
        
        return result
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to readable timestamp (HH:MM:SS or MM:SS)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _generate_id(self, asset: str, value: float, date: Optional[str]) -> str:
        """Generate unique prediction ID"""
        import hashlib
        data = f"{asset}_{value}_{date}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]