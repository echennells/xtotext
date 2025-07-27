"""
Optimized LLM-based prediction extractor with reduced memory/IO footprint
"""
from typing import List, Dict, Optional, Generator
from datetime import datetime
import json
import gc
import time

from llm.llm_client import OpenAIClient
from .models import Prediction, PredictionType, Confidence, TimeFrame


class OptimizedChunkProcessor:
    """Memory-efficient chunk processor using generators"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        # Import config to get dynamic chunk size based on model
        import config
        
        # Use provided values or defaults from config
        self.chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
        self.overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    
    def create_chunks_generator(self, text: str) -> Generator[Dict[str, any], None, None]:
        """
        Generate chunks one at a time to avoid creating all chunks in memory
        
        Yields:
            Chunk dictionaries one at a time
        """
        text_length = len(text)
        start = 0
        chunk_num = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Find sentence boundary near the end of chunk
            if end < text_length:
                # Look for delimiter in last 500 chars of the chunk
                search_start = max(start, end - 500)
                delimiter_found = False
                for delimiter in ['. ', '? ', '! ', '\n\n', '\n']:
                    last_delimiter = text.rfind(delimiter, search_start, end)
                    # Only use delimiter if it doesn't make chunk too small
                    if last_delimiter > start and (last_delimiter - start) > self.chunk_size * 0.8:
                        end = last_delimiter + len(delimiter)
                        delimiter_found = True
                        break
                
                # If no good delimiter found, just use the full chunk size
                if not delimiter_found:
                    end = min(start + self.chunk_size, text_length)
            
            yield {
                'text': text[start:end],
                'chunk_num': chunk_num,
                'start_char': start,
                'end_char': end,
                'is_first': chunk_num == 0,
                'is_last': end >= text_length
            }
            
            # Break if we've reached the end
            if end >= text_length:
                break
                
            start = end - self.overlap
            chunk_num += 1


class OptimizedLLMPredictionExtractor:
    """Memory-optimized prediction extractor"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.llm_client = OpenAIClient(api_key)
        self.chunk_processor = OptimizedChunkProcessor()
        
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
        }
    
    def extract_predictions_from_file(self, transcript_file: str, episode_info: Dict) -> List[Prediction]:
        # Store transcript data for timestamp recovery
        self._current_transcript_data = None
        """
        Extract predictions directly from file without loading entire JSON into memory
        
        Args:
            transcript_file: Path to transcript JSON file
            episode_info: Episode metadata
            
        Returns:
            List of Prediction objects
        """
        # Stream the JSON file to extract just the text field
        text = self._extract_text_from_json_stream(transcript_file)
        
        # Load segments for timestamp recovery (store temporarily)
        segments = self._load_segments_only(transcript_file)
        self._current_transcript_data = {'segments': segments}
        print(f"[TIMESTAMP RECOVERY] Loaded {len(segments)} segments for timestamp recovery")
        
        # Process with optimized chunking
        predictions = self.extract_predictions_optimized(text, episode_info)
        
        # Match predictions to segments
        print(f"\n[TIMESTAMP RECOVERY] Matching {len(predictions)} predictions to segments...")
        for i, prediction in enumerate(predictions):
            if prediction.raw_text and segments:
                print(f"\n[MATCH {i+1}] Searching for: '{prediction.raw_text[:50]}...'")
                segment_info = self._find_quote_in_segments(prediction.raw_text, segments)
                if segment_info:
                    prediction.timestamp_start = segment_info['start']
                    prediction.timestamp_end = segment_info['end']
                    prediction.timestamp = prediction.format_timestamp(segment_info['start'])
                    print(f"[MATCH {i+1}] ✓ Found at {prediction.timestamp}")
                    
                    video_id = episode_info.get('video_id')
                    if video_id:
                        prediction.youtube_link = prediction.generate_youtube_link(video_id)
                else:
                    print(f"[MATCH {i+1}] ✗ Not found in segments")
        
        # Clear segments and transcript data from memory
        del segments
        self._current_transcript_data = None
        gc.collect()
        
        return predictions
    
    def _extract_text_from_json_stream(self, filepath: str) -> str:
        """Extract just the 'text' field from JSON without loading entire file"""
        import ijson
        
        with open(filepath, 'rb') as file:
            parser = ijson.items(file, 'text')
            for text in parser:
                return text
        return ""
    
    def _load_segments_only(self, filepath: str) -> List[Dict]:
        """Load only the segments array from JSON"""
        import ijson
        
        with open(filepath, 'rb') as file:
            parser = ijson.items(file, 'segments.item')
            segments = []
            for segment in parser:
                # Only keep essential fields to save memory
                segments.append({
                    'start': segment.get('start'),
                    'end': segment.get('end'),
                    'text': segment.get('text')
                })
            return segments
    
    def extract_predictions_optimized(self, text: str, episode_info: Dict) -> List[Prediction]:
        """
        Extract predictions using memory-efficient chunking
        
        Args:
            text: Full transcript text
            episode_info: Episode metadata
            
        Returns:
            List of Prediction objects
        """
        chunk_threshold = 80000  # Increased threshold since we're more memory efficient
        
        if len(text) > chunk_threshold:
            print(f"\n=== OPTIMIZED CHUNKING ===")
            print(f"Text length: {len(text):,} characters")
            print(f"Chunk size: {self.chunk_processor.chunk_size:,} chars")
            print(f"Chunk overlap: {self.chunk_processor.overlap:,} chars")
            
            # Calculate expected chunks
            expected_chunks = (len(text) - self.chunk_processor.overlap) // (self.chunk_processor.chunk_size - self.chunk_processor.overlap) + 1
            print(f"Expected chunks: ~{expected_chunks}")
            
            all_predictions = []
            seen_predictions = set()
            
            # Process chunks one at a time using generator
            chunk_count = 0
            for chunk in self.chunk_processor.create_chunks_generator(text):
                chunk_count += 1
                context = {
                    'episode_title': episode_info.get('title', ''),
                    'episode_date': episode_info.get('date', ''),
                    'chunk_position': f"{chunk['chunk_num'] + 1}"
                }
                
                print(f"\nProcessing chunk {chunk_count} (chars {chunk['start_char']:,}-{chunk['end_char']:,}, size: {chunk['end_char'] - chunk['start_char']:,})...")
                
                # Extract predictions for this chunk
                predictions = self.llm_client.extract_predictions(
                    chunk['text'],
                    context
                )
                
                # Process predictions immediately and clear chunk from memory
                for pred in predictions:
                    # Safe price parsing
                    try:
                        price_val = float(pred.get('price', 0))
                    except (ValueError, TypeError):
                        price_val = 0
                    
                    key = (
                        pred.get('asset', '').upper(),
                        price_val,
                        pred.get('timeframe', '').lower()
                    )
                    
                    if key not in seen_predictions and key[1] > 0:
                        seen_predictions.add(key)
                        prediction_obj = self._create_prediction_object(pred, episode_info)
                        if prediction_obj:
                            all_predictions.append(prediction_obj)
                
                # Clear chunk from memory
                del chunk
                gc.collect()
                
                # Add delay between API calls to avoid rate limiting
                time.sleep(1.0)  # 1 second delay between chunks
            
            print(f"\nTotal unique predictions: {len(all_predictions)}")
            return all_predictions
            
        else:
            # Small enough to process in one shot
            print(f"Processing entire transcript ({len(text)} chars) in one request...")
            context = {
                'episode_title': episode_info.get('title', ''),
                'episode_date': episode_info.get('date', ''),
            }
            
            predictions = self.llm_client.extract_predictions(text, context)
            
            prediction_objects = []
            for pred_data in predictions:
                prediction = self._create_prediction_object(pred_data, episode_info)
                if prediction:
                    prediction_objects.append(prediction)
            
            return prediction_objects
    
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
    
    def _recover_missing_timestamp(self, pred_data: Dict, episode_info: Dict) -> Optional[str]:
        """Try to recover missing timestamp by searching the transcript"""
        quote = pred_data.get('quote', '').strip()
        # If no quote, try using context
        if not quote:
            quote = pred_data.get('context', '').strip()
        
        print(f"   [RECOVERY] Quote to search: '{quote[:60]}...'")
        
        if not quote:
            print(f"   [RECOVERY] No quote or context available")
            return None
            
        if not self._current_transcript_data:
            print(f"   [RECOVERY] No transcript data loaded")
            return None
            
        print(f"   [RECOVERY] Searching in {len(self._current_transcript_data.get('segments', []))} segments...")
        
        # Try to find the quote in segments
        segments = self._current_transcript_data.get('segments', [])
        if segments:
            match = self._find_quote_in_segments(quote, segments)
            if match:
                timestamp = self._format_timestamp(match['start'])
                print(f"   [RECOVERY] ✓ Found at timestamp: {timestamp} (segment {match.get('segment_index', 'unknown')})")
                return timestamp
            else:
                print(f"   [RECOVERY] ✗ Could not find exact quote in transcript")
                # Try searching for key phrases
                key_words = quote.lower().split()[:5]  # First 5 words
                print(f"   [RECOVERY] Trying partial match with keywords: {' '.join(key_words)}")
                # Search for partial matches
                for i, segment in enumerate(segments):
                    seg_text = segment.get('text', '').lower()
                    if all(word in seg_text for word in key_words[:3]):  # At least first 3 words
                        timestamp = self._format_timestamp(segment.get('start', 0))
                        print(f"   [RECOVERY] ✓ Found partial match at: {timestamp} (segment {i})")
                        return timestamp
                
                print(f"   [RECOVERY] ✗ No partial matches found either")
        
        return None
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to timestamp string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _create_prediction_object(self, pred_data: Dict, episode_info: Dict) -> Optional[Prediction]:
        """Convert LLM output to Prediction object"""
        try:
            asset_raw = pred_data.get('asset', '').lower()
            asset = self.asset_map.get(asset_raw, asset_raw.upper())
            
            if not asset or len(asset) > 10:
                return None
            
            price_str = str(pred_data.get('price', '0'))
            
            # Handle price ranges
            if '-' in price_str and any(c.isdigit() for c in price_str):
                import re
                numbers = re.findall(r'(\d+)', price_str)
                if len(numbers) >= 2:
                    low = float(numbers[0])
                    high = float(numbers[1])
                    if 'k' in price_str.lower():
                        low *= 1000
                        high *= 1000
                    price = (low + high) / 2
                else:
                    price = 0
            else:
                try:
                    # Remove 'k' and handle as number
                    cleaned_price = price_str.lower().replace('k', '').replace(',', '')
                    price = float(cleaned_price)
                    if 'k' in price_str.lower():
                        price *= 1000
                except:
                    price = 0
            
            if price <= 0:
                return None
            
            # Fix common price interpretation issues
            if asset == 'BTC' and price < 1000:
                price = price * 1000
            
            if asset == 'BTC' and 10 <= price <= 20:
                price = (price + 100) * 1000
            
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
            
            # Get timestamp - try from pred_data first, then recover if missing
            timestamp = pred_data.get('timestamp')
            if not timestamp:
                print(f"\n[TIMESTAMP RECOVERY] Missing timestamp for {asset} ${price}")
                # Try to recover missing timestamp
                timestamp = self._recover_missing_timestamp(pred_data, episode_info)
                if timestamp:
                    print(f"[TIMESTAMP RECOVERY] ✓ Recovered: {timestamp}")
                else:
                    print(f"[TIMESTAMP RECOVERY] ✗ Failed to recover timestamp")
            
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
                timestamp=timestamp,
                raw_text=pred_data.get('quote', '')[:500],
                prediction_id=self._generate_id(asset, price, timeframe_data.get('predicted_date'))
            )
            
            return prediction
            
        except (ValueError, KeyError, TypeError) as e:
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
                    try:
                        current_date = datetime.fromisoformat(episode_date)
                    except:
                        current_date = datetime.now()
                    
                    year = current_date.year
                    if month_num < current_date.month:
                        year += 1  # Next year
                    
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
        
        elif 'end of year' in timeframe_lower or 'eoy' in timeframe_lower or 'this year' in timeframe_lower:
            result['time_frame'] = TimeFrame.EOY
            try:
                current_date = datetime.fromisoformat(episode_date)
            except:
                current_date = datetime.now()
            result['predicted_date'] = f"{current_date.year}-12-31"
        
        elif 'month' in timeframe_lower:
            # Extract number of months
            import re
            num_match = re.search(r'(\d+)\s*month', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = int(num_match.group(1))
            elif 'next month' in timeframe_lower:
                result['time_frame'] = TimeFrame.MONTHS
                result['time_value'] = 1
        
        elif 'week' in timeframe_lower:
            # Extract number of weeks
            import re
            num_match = re.search(r'(\d+)\s*week', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = int(num_match.group(1))
            elif 'this week' in timeframe_lower:
                result['time_frame'] = TimeFrame.WEEKS
                result['time_value'] = 0  # current week
        
        elif 'day' in timeframe_lower:
            # Extract number of days
            import re
            num_match = re.search(r'(\d+)\s*day', timeframe_lower)
            if num_match:
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = int(num_match.group(1))
            elif 'today' in timeframe_lower:
                result['time_frame'] = TimeFrame.DAYS
                result['time_value'] = 0
        
        # Handle specific date patterns like "on July 18th"
        elif ' on ' in timeframe_lower:
            import re
            # Try to parse "on Month Day"
            date_match = re.search(r'on\s+(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?', timeframe_lower)
            if date_match:
                month_str = date_match.group(1).lower()
                day = int(date_match.group(2))
                
                month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                              'july', 'august', 'september', 'october', 'november', 'december']
                if month_str in month_names:
                    month_num = month_names.index(month_str) + 1
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
        
        # Log what we parsed for debugging
        if not result:
            print(f"Warning: Could not parse timeframe: '{timeframe_str}'")
        
        return result
    
    def _generate_id(self, asset: str, value: float, date: Optional[str]) -> str:
        """Generate unique prediction ID"""
        import hashlib
        data = f"{asset}_{value}_{date}_{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]