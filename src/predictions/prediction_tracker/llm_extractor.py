"""
LLM-based prediction extractor for crypto podcasts
"""
from typing import List, Dict, Optional
from datetime import datetime
import json

from llm.llm_client import OpenAIClient, ChunkProcessor
from .models import Prediction, PredictionType, Confidence, TimeFrame


class LLMPredictionExtractor:
    """Extract predictions using LLM for better context understanding"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.llm_client = OpenAIClient(api_key)
        self.chunk_processor = ChunkProcessor()
        
        # Asset normalization mapping
        self.asset_map = {
            # Cryptocurrencies
            'bitcoin': 'BTC', 'btc': 'BTC',
            'ethereum': 'ETH', 'eth': 'ETH', 'ether': 'ETH',
            'smlr': 'SMLR', 'semler': 'SMLR', 'similar': 'SMLR',  # Common transcription error
            'solana': 'SOL', 'sol': 'SOL',
            'chainlink': 'LINK', 'link': 'LINK',
            'cardano': 'ADA', 'ada': 'ADA',
            'ripple': 'XRP', 'xrp': 'XRP',
            'doge': 'DOGE', 'dogecoin': 'DOGE',
            'matic': 'MATIC', 'polygon': 'MATIC',
            
            # Crypto-related stocks
            'mstr': 'MSTR', 'microstrategy': 'MSTR', 'micro strategy': 'MSTR', 'misty': 'MSTR',
            'coin': 'COIN', 'coinbase': 'COIN',
            'mara': 'MARA', 'marathon': 'MARA',
            'riot': 'RIOT',
            'clsk': 'CLSK', 'cleanspark': 'CLSK',
            'hut': 'HUT', 'hut8': 'HUT',
            'bitf': 'BITF', 'bitfarms': 'BITF',
            'hive': 'HIVE',
            'gbtc': 'GBTC', 'grayscale': 'GBTC',
        }
    
    def extract_predictions_with_timestamps(self, transcript_data: Dict, episode_info: Dict) -> List[Prediction]:
        """
        Extract predictions from transcript with timestamp information
        
        Args:
            transcript_data: Full transcript data with segments
            episode_info: Episode metadata (title, date, etc.)
            
        Returns:
            List of Prediction objects with timestamps
        """
        text = transcript_data.get('text', '')
        segments = transcript_data.get('segments', [])
        
        # Extract predictions from text
        predictions = self.extract_predictions(text, episode_info)
        
        # Match predictions to segments for timestamps
        for prediction in predictions:
            if prediction.raw_text:
                # Find the segment(s) containing this quote
                segment_info = self._find_quote_in_segments(prediction.raw_text, segments)
                if segment_info:
                    prediction.timestamp_start = segment_info['start']
                    prediction.timestamp_end = segment_info['end']
                    prediction.timestamp = prediction.format_timestamp(segment_info['start'])
                    
                    # Generate YouTube link if we have video ID
                    video_id = episode_info.get('video_id')
                    if video_id:
                        prediction.youtube_link = prediction.generate_youtube_link(video_id)
        
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
            
            # Simple matching - check if significant portion of quote is in segment
            matching_words = sum(1 for word in quote_words if word in segment_text)
            score = matching_words / len(quote_words) if quote_words else 0
            
            if score > best_score and score > 0.5:  # At least 50% match
                best_score = score
                best_match = {
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'segment_index': i
                }
                
                # Check if quote spans multiple segments
                if score < 0.9 and i < len(segments) - 1:
                    # Check next segment too
                    next_segment_text = segments[i + 1].get('text', '').lower()
                    combined_text = segment_text + " " + next_segment_text
                    combined_score = sum(1 for word in quote_words if word in combined_text) / len(quote_words)
                    
                    if combined_score > 0.8:
                        best_match['end'] = segments[i + 1].get('end', best_match['end'])
        
        return best_match
    
    def extract_predictions(self, text: str, episode_info: Dict) -> List[Prediction]:
        """
        Extract predictions from transcript using LLM
        
        Args:
            text: Full transcript text
            episode_info: Episode metadata (title, date, etc.)
            
        Returns:
            List of Prediction objects
        """
        # Check text length - GPT-4o-mini has 128K context
        # Roughly 1 token = 0.75 words, so 128K tokens ≈ 96K words ≈ 480K characters
        # Lower threshold for better accuracy
        chunk_threshold = 50000  # Much lower threshold to ensure quality
        
        if len(text) > chunk_threshold:
            print(f"\n=== CHUNKING DETAILS ===")
            print(f"Text length: {len(text):,} characters")
            print(f"Chunking threshold: {chunk_threshold:,} characters")
            print(f"Chunk size: {self.chunk_processor.chunk_size:,} chars")
            print(f"Chunk overlap: {self.chunk_processor.overlap:,} chars")
            
            # Fall back to chunking for long texts
            chunks = self.chunk_processor.create_chunks(text)
            print(f"Created {len(chunks)} chunks")
            
            chunk_predictions = []
            for i, chunk in enumerate(chunks):
                context = {
                    'episode_title': episode_info.get('title', ''),
                    'episode_date': episode_info.get('date', ''),
                    'chunk_position': f"{chunk['chunk_num'] + 1} of {len(chunks)}"
                }
                
                print(f"\nProcessing chunk {i+1}/{len(chunks)}:")
                print(f"  Characters {chunk['start_char']:,} to {chunk['end_char']:,}")
                print(f"  Chunk text preview: {chunk['text'][:100]}...")
                
                predictions = self.llm_client.extract_predictions(
                    chunk['text'],
                    context
                )
                
                print(f"  Found {len(predictions)} predictions in this chunk")
                for pred in predictions:
                    print(f"    - {pred.get('asset', 'Unknown')} to ${pred.get('price', 0):,.0f}")
                
                chunk_predictions.append(predictions)
            
            print(f"\n=== MERGING PREDICTIONS ===")
            print(f"Total predictions before merge: {sum(len(p) for p in chunk_predictions)}")
            merged_predictions = self.chunk_processor.merge_predictions(chunk_predictions)
            print(f"Total predictions after merge: {len(merged_predictions)}")
            print("=" * 40)
        else:
            # Process entire transcript in one shot
            print(f"Processing entire transcript ({len(text)} chars) in one request...")
            context = {
                'episode_title': episode_info.get('title', ''),
                'episode_date': episode_info.get('date', ''),
            }
            
            merged_predictions = self.llm_client.extract_predictions(text, context)
            
            print(f"\n=== EXTRACTION RESULTS ===")
            print(f"Found {len(merged_predictions)} predictions")
            for pred in merged_predictions:
                print(f"  - {pred.get('asset', 'Unknown')} to ${pred.get('price', 0):,.0f}")
            print("=" * 40)
        
        # Convert to Prediction objects
        prediction_objects = []
        for pred_data in merged_predictions:
            prediction = self._create_prediction_object(pred_data, episode_info)
            if prediction:
                prediction_objects.append(prediction)
        
        return prediction_objects
    
    def _create_prediction_object(
        self, 
        pred_data: Dict, 
        episode_info: Dict
    ) -> Optional[Prediction]:
        """
        Convert LLM output to Prediction object
        
        Args:
            pred_data: Prediction data from LLM
            episode_info: Episode metadata
            
        Returns:
            Prediction object or None if invalid
        """
        try:
            # Normalize asset
            asset_raw = pred_data.get('asset', '').lower()
            asset = self.asset_map.get(asset_raw, asset_raw.upper())
            
            # Validate asset
            if not asset or len(asset) > 10:
                return None
            
            # Parse price (handle ranges like "60-80k")
            price_str = str(pred_data.get('price', '0'))
            
            # Handle price ranges
            if '-' in price_str and any(c.isdigit() for c in price_str):
                # Extract numbers from range (e.g., "60-80k" -> 70)
                import re
                numbers = re.findall(r'(\d+)', price_str)
                if len(numbers) >= 2:
                    # Use midpoint of range
                    low = float(numbers[0])
                    high = float(numbers[1])
                    # Check if 'k' is in the string (thousands)
                    if 'k' in price_str.lower():
                        low *= 1000
                        high *= 1000
                    price = (low + high) / 2
                else:
                    price = 0
            else:
                # Regular single price
                try:
                    # Remove 'k' suffix if present
                    if 'k' in price_str.lower():
                        price = float(price_str.lower().replace('k', '')) * 1000
                    else:
                        price = float(price_str)
                except:
                    price = 0
            
            if price <= 0:
                return None
            
            # Fix common price interpretation issues
            # For BTC, prices under 1000 are likely in thousands
            if asset == 'BTC' and price < 1000:
                price = price * 1000  # Convert to full dollar amount
            
            # Special case: "14-ish" for BTC means $114k not $14k
            if asset == 'BTC' and 10 <= price <= 20:
                price = (price + 100) * 1000  # 14 -> 114,000
            
            # Parse timeframe
            timeframe_data = self._parse_timeframe(
                pred_data.get('timeframe', ''),
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
            
            # Create prediction
            prediction = Prediction(
                asset=asset,
                prediction_type=PredictionType.PRICE_TARGET,
                value=price,
                predicted_date=timeframe_data.get('predicted_date'),
                time_frame=timeframe_data.get('time_frame'),
                time_value=timeframe_data.get('time_value'),
                speaker=pred_data.get('speaker'),
                confidence=confidence,
                reasoning=pred_data.get('reasoning'),
                episode=episode_info.get('title', ''),
                episode_date=episode_info.get('date', ''),
                timestamp=episode_info.get('timestamp'),
                raw_text=pred_data.get('quote', '')[:500],  # Limit quote length
                prediction_id=self._generate_id(asset, price, timeframe_data.get('predicted_date'))
            )
            
            return prediction
            
        except (ValueError, KeyError, TypeError) as e:
            # Invalid prediction data
            return None
    
    def _parse_timeframe(self, timeframe_str: str, episode_date: str) -> Dict:
        """
        Parse timeframe string into structured data
        
        Args:
            timeframe_str: Timeframe from LLM (e.g., "by August 1st", "in 3 months")
            episode_date: Date of episode for relative calculations
            
        Returns:
            Dict with predicted_date, time_frame, and time_value
        """
        timeframe_lower = timeframe_str.lower()
        result = {}
        
        # Try to parse specific dates
        if any(month in timeframe_lower for month in [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]):
            # Extract month and possibly day/year
            # This is simplified - you might want to use dateutil.parser
            for month_num, month_name in enumerate([
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december'
            ], 1):
                if month_name in timeframe_lower:
                    # Try to extract day
                    import re
                    day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?', timeframe_lower)
                    day = int(day_match.group(1)) if day_match else 1
                    
                    # Determine year
                    current_date = datetime.fromisoformat(episode_date)
                    year = current_date.year
                    if month_num < current_date.month:
                        year += 1  # Next year
                    
                    predicted_date = f"{year}-{month_name}-{day:02d}"
                    result['predicted_date'] = predicted_date
                    result['time_frame'] = TimeFrame.SPECIFIC_DATE
                    break
        
        # Relative timeframes
        elif 'tomorrow' in timeframe_lower:
            current_date = datetime.fromisoformat(episode_date)
            next_day = current_date.replace(day=current_date.day + 1)
            result['predicted_date'] = next_day.isoformat()[:10]
            result['time_frame'] = TimeFrame.SPECIFIC_DATE
        
        elif 'end of year' in timeframe_lower or 'eoy' in timeframe_lower:
            result['time_frame'] = TimeFrame.EOY
        
        elif 'month' in timeframe_lower:
            # Extract number of months
            import re
            num_match = re.search(r'(\d+)\s*month', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = int(num_match.group(1))
        
        elif 'week' in timeframe_lower:
            # Extract number of weeks
            import re
            num_match = re.search(r'(\d+)\s*week', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = int(num_match.group(1))
        
        elif 'day' in timeframe_lower:
            # Extract number of days
            import re
            num_match = re.search(r'(\d+)\s*day', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = int(num_match.group(1))
        
        return result
    
    def _generate_id(self, asset: str, value: float, date: Optional[str]) -> str:
        """Generate unique prediction ID"""
        import hashlib
        data = f"{asset}_{value}_{date}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]