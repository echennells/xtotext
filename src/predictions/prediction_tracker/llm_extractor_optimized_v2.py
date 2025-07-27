"""
Optimized LLM prediction extractor with improved context handling
"""
import gc
import time
import json
from typing import List, Dict, Optional
from pathlib import Path

from predictions.prediction_tracker.models import Prediction, PredictionType, Confidence, TimeFrame
from llm.llm_client import OpenAIClient


class ImprovedChunkProcessor:
    """Process large texts with better context preservation"""
    
    def __init__(self, chunk_size: int = 400000, context_before: int = 20000):
        """
        Initialize with improved context handling
        
        Args:
            chunk_size: Size of main content chunk (chars)
            context_before: Additional context to include before chunk (chars)
        """
        self.chunk_size = chunk_size
        self.context_before = context_before  # ~5k tokens of preceding context
        
    def create_chunks_with_context(self, text: str):
        """
        Create chunks with substantial preceding context
        
        This ensures that ticker mentions from earlier in the conversation
        are included when processing price predictions
        """
        text_length = len(text)
        
        # For first chunk, start at beginning
        start = 0
        chunk_num = 0
        
        while start < text_length:
            # Calculate context start (how far back to include)
            context_start = max(0, start - self.context_before)
            
            # Calculate chunk end
            end = min(start + self.chunk_size, text_length)
            
            # Find sentence boundary near the end
            if end < text_length:
                search_start = max(start, end - 500)
                for delimiter in ['. ', '? ', '! ', '\n\n', '\n']:
                    last_delimiter = text.rfind(delimiter, search_start, end)
                    if last_delimiter > start:
                        end = last_delimiter + len(delimiter)
                        break
            
            # Include context before + main chunk
            chunk_text = text[context_start:end]
            
            # Mark where the "new" content starts in this chunk
            new_content_offset = start - context_start
            
            yield {
                'text': chunk_text,
                'chunk_num': chunk_num,
                'context_start': context_start,
                'main_start': start,
                'end': end,
                'new_content_offset': new_content_offset,
                'is_first': chunk_num == 0,
                'is_last': end >= text_length
            }
            
            if end >= text_length:
                break
                
            # Next chunk starts where this one's main content ended
            # (but will include context from before that point)
            start = end
            chunk_num += 1


class OptimizedLLMPredictionExtractorV2:
    """Prediction extractor with improved context awareness"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.llm_client = OpenAIClient(api_key)
        self.chunk_processor = ImprovedChunkProcessor()
        
        # Asset normalization mapping
        self.asset_map = {
            'bitcoin': 'BTC', 'btc': 'BTC',
            'ethereum': 'ETH', 'eth': 'ETH', 'ether': 'ETH',
            'smlr': 'SMLR', 'semler': 'SMLR', 'similar': 'SMLR',
            'solana': 'SOL', 'sol': 'SOL',
            'chainlink': 'LINK', 'link': 'LINK',
            'cardano': 'ADA', 'ada': 'ADA',
            'ripple': 'XRP', 'xrp': 'XRP',
            'doge': 'DOGE', 'dogecoin': 'DOGE',
            'matic': 'MATIC', 'polygon': 'MATIC',
            'mstr': 'MSTR', 'microstrategy': 'MSTR', 'micro strategy': 'MSTR', 'misty': 'MSTR',
            'coin': 'COIN', 'coinbase': 'COIN',
            'mara': 'MARA', 'marathon': 'MARA',
            'riot': 'RIOT',
            'clsk': 'CLSK', 'cleanspark': 'CLSK',
            'hut': 'HUT', 'hut8': 'HUT',
            'bitf': 'BITF', 'bitfarms': 'BITF',
            'hive': 'HIVE',
            'gbtc': 'GBTC', 'grayscale': 'GBTC',
            # Add Meta Planet
            'meta planet': 'METAP', 'metaplanet': 'METAP', 'meta': 'METAP',
        }
    
    def extract_predictions_with_context(self, text: str, episode_info: Dict) -> List[Prediction]:
        """
        Extract predictions with improved context handling
        """
        # Check if we need to chunk
        if len(text) <= self.chunk_processor.chunk_size:
            # Process in one shot
            return self._process_single_chunk(text, episode_info)
        
        # Process in chunks with context
        all_predictions = []
        seen_predictions = set()
        
        for chunk_info in self.chunk_processor.create_chunks_with_context(text):
            chunk_num = chunk_info['chunk_num']
            chunk_text = chunk_info['text']
            new_content_offset = chunk_info['new_content_offset']
            
            print(f"\nProcessing chunk {chunk_num + 1}:")
            print(f"  Total size: {len(chunk_text)} chars")
            print(f"  New content starts at: {new_content_offset}")
            
            # Extract predictions from chunk
            context = {
                'episode_title': episode_info.get('title', ''),
                'episode_date': episode_info.get('date', ''),
                'chunk_num': chunk_num,
                'is_first_chunk': chunk_info['is_first'],
                'is_last_chunk': chunk_info['is_last'],
                'new_content_offset': new_content_offset,
                'instruction': 'Focus on predictions in the NEW content (after the marked offset), but use the full context to understand what assets are being discussed.'
            }
            
            # Modify the prompt to be aware of the context structure
            predictions = self._extract_with_context_awareness(chunk_text, context)
            
            # Deduplicate
            for pred in predictions:
                key = (
                    pred.get('asset', '').upper(),
                    float(pred.get('price', 0)),
                    pred.get('quote', '').lower()[:50]  # Use partial quote for dedup
                )
                
                if key not in seen_predictions and key[1] > 0:
                    seen_predictions.add(key)
                    prediction_obj = self._create_prediction_object(pred, episode_info)
                    if prediction_obj:
                        all_predictions.append(prediction_obj)
            
            # Clean up
            del chunk_text
            gc.collect()
            time.sleep(1.0)
        
        return all_predictions
    
    def _extract_with_context_awareness(self, text: str, context: Dict) -> List[Dict]:
        """
        Extract predictions with awareness of context structure
        """
        # Create a modified system prompt that understands the context structure
        system_prompt = """You are a financial prediction extractor for the Bitcoin Dive Bar podcast. Extract price predictions for cryptocurrencies and crypto-related stocks.

IMPORTANT: This text includes CONTEXT from earlier in the conversation followed by NEW CONTENT. 
The context helps you understand what assets are being discussed, but you should ONLY extract predictions from the NEW CONTENT section.

CRITICAL CONTEXT:
- This is a casual, informal podcast where speakers use slang and shorthand
- Numbers are often expressed informally (e.g., "120" = $120,000 for BTC, "200K" = $200,000)
- Speakers often discuss an asset/ticker, then mention prices much later
- Pay attention to what asset is being discussed in the conversation flow
- Common tickers: BTC, ETH, SMLR, MSTR, COIN, MARA, Meta Planet, etc.

Extract ONLY clear FUTURE price predictions that include:
1. An asset (properly identified from context)
2. A specific price target
3. Optional timeframe

IMPORTANT: When you see a price mentioned, look back in the context to determine which asset is being discussed. Don't assume it's the last mentioned ticker.

For each prediction, extract:
- asset: The correct ticker symbol based on conversation context
- price: The predicted price
- timeframe: When the prediction is for (if mentioned)
- speaker: Who made the prediction
- confidence: high/medium/low based on tone
- quote: The exact quote containing the prediction
- reasoning: Any reasoning provided

Output as JSON with a "predictions" array."""

        # Add context information to the user prompt
        new_content_offset = context.get('new_content_offset', 0)
        
        user_prompt = f"""Extract predictions from this transcript excerpt.

CONTEXT STRUCTURE:
- First {new_content_offset} characters are PREVIOUS CONTEXT (for understanding what assets are discussed)
- After character {new_content_offset} is NEW CONTENT to extract predictions from

Text:
{text}

Only extract predictions that appear in the NEW CONTENT section (after character {new_content_offset}), but use the full text to understand which asset is being discussed.

Episode info: {json.dumps(context)}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.llm_client.extract_predictions_raw(messages)
    
    def _create_prediction_object(self, pred_data: Dict, episode_info: Dict) -> Optional[Prediction]:
        """Convert LLM output to Prediction object"""
        try:
            asset_raw = pred_data.get('asset', '').lower()
            asset = self.asset_map.get(asset_raw, asset_raw.upper())
            
            if not asset or len(asset) > 10:
                return None
                
            value = float(pred_data.get('price', 0))
            if value <= 0:
                return None
            
            prediction = Prediction(
                asset=asset,
                prediction_type=PredictionType.PRICE_TARGET,
                value=value,
                speaker=pred_data.get('speaker', 'Unknown'),
                confidence=self._parse_confidence(pred_data.get('confidence', 'medium')),
                episode=episode_info.get('title', ''),
                episode_date=episode_info.get('date', ''),
                raw_text=pred_data.get('quote', ''),
                reasoning=pred_data.get('reasoning'),
                timestamp=None  # Will be set later
            )
            
            # Handle timeframe
            timeframe_str = pred_data.get('timeframe', '').lower()
            if any(term in timeframe_str for term in ['day', 'today', 'tomorrow']):
                prediction.time_frame = TimeFrame.DAYS
            elif any(term in timeframe_str for term in ['week']):
                prediction.time_frame = TimeFrame.WEEKS
            elif any(term in timeframe_str for term in ['month']):
                prediction.time_frame = TimeFrame.MONTHS
            elif any(term in timeframe_str for term in ['year']):
                prediction.time_frame = TimeFrame.YEARS
            
            return prediction
            
        except Exception as e:
            print(f"Error creating prediction: {e}")
            return None
    
    def _parse_confidence(self, confidence_str: str) -> Confidence:
        """Parse confidence string to enum"""
        confidence_map = {
            'high': Confidence.HIGH,
            'medium': Confidence.MEDIUM,
            'low': Confidence.LOW,
            'uncertain': Confidence.LOW
        }
        return confidence_map.get(confidence_str.lower(), Confidence.MEDIUM)