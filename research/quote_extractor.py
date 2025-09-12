"""
Quote Extractor with YouTube Timestamp Links
Extracts key quotable moments and creates direct YouTube links
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import OPENROUTER_API_KEY


@dataclass
class Quote:
    """Represents a quotable moment with metadata"""
    text: str
    speaker: Optional[str]
    topic: str
    importance: str  # "critical", "high", "medium"
    start_time: float  # seconds
    end_time: float
    youtube_link: str
    file_source: str
    context: Optional[str] = None  # surrounding context
    
    @property
    def formatted_time(self) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(self.start_time // 60)
        seconds = int(self.start_time % 60)
        return f"{minutes:02d}:{seconds:02d}"


class QuoteExtractor:
    """Extract key quotes with YouTube timestamps"""
    
    YOUTUBE_BASE_URL = "https://www.youtube.com/watch?v=gWWxDd3mhZc"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.quotes: List[Quote] = []
    
    def find_quote_in_segments(self, quote_text: str, segments: List[Dict]) -> Optional[Tuple[float, float]]:
        """Find a quote in transcript segments and return timestamp"""
        quote_lower = quote_text.lower()[:100]  # Use first 100 chars for matching
        
        for segment in segments:
            segment_text = segment.get('text', '').lower()
            if quote_lower[:50] in segment_text or segment_text[:50] in quote_lower:
                return segment['start'], segment['end']
        
        # Fuzzy match - look for key words
        quote_words = set(quote_lower.split()[:10])
        best_match = None
        best_score = 0
        
        for segment in segments:
            segment_text = segment.get('text', '').lower()
            segment_words = set(segment_text.split())
            score = len(quote_words.intersection(segment_words))
            
            if score > best_score and score > len(quote_words) * 0.6:
                best_score = score
                best_match = (segment['start'], segment['end'])
        
        return best_match
    
    def extract_key_quotes(self, transcript_file: Path) -> List[Quote]:
        """Extract key quotes from a single transcript file"""
        # Load raw transcript with timestamps
        with open(transcript_file, 'r') as f:
            data = json.load(f)
        
        text = data.get('text', '')
        segments = data.get('segments', [])
        
        # Also check for processed version for cleaner text
        processed_path = transcript_file.parent / transcript_file.name.replace('_transcript.json', '_transcript_claude_postprocessed.json')
        if processed_path.exists():
            with open(processed_path, 'r') as f:
                processed_data = json.load(f)
                clean_text = processed_data.get('text', text)
        else:
            clean_text = text
        
        # Use LLM to identify key quotes
        prompt = """Identify the 5-8 most important, quotable moments from this technical conference transcript.

Look for:
1. BREAKTHROUGH MOMENTS - "We realized BitVM2 is completely obsolete"
2. TECHNICAL INSIGHTS - Clear explanations of complex concepts
3. PERFORMANCE METRICS - "1000x improvement", "from 6MB to 70KB"
4. FUTURE PREDICTIONS - What's coming next
5. MEMORABLE ANALOGIES - Ways complex ideas are explained simply
6. CHALLENGE STATEMENTS - Problems that need solving

For each quote:
- Extract the EXACT words (30-100 words ideal)
- Identify the topic (BitVM, Garbled Circuits, SNARK, etc.)
- Rate importance (critical/high/medium)
- Note the speaker if identifiable

Output as JSON:
{
  "quotes": [
    {
      "text": "exact quote here",
      "topic": "topic",
      "importance": "critical/high/medium",
      "speaker": "name or null",
      "context": "what was being discussed"
    }
  ]
}

Transcript:
""" + clean_text[:50000]  # Use first 50k chars
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek/deepseek-v3-base",
                messages=[
                    {"role": "system", "content": "Extract memorable technical quotes from conference talks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048
            )
            
            result = json.loads(response.choices[0].message.content)
            quotes_data = result.get('quotes', [])
            
        except Exception as e:
            print(f"Error extracting quotes: {e}")
            # Fallback to pattern matching
            quotes_data = self.pattern_based_extraction(clean_text)
        
        # Match quotes to timestamps and create Quote objects
        extracted_quotes = []
        for quote_data in quotes_data:
            quote_text = quote_data.get('text', '')
            
            # Find timestamp for this quote
            timestamp = self.find_quote_in_segments(quote_text, segments)
            
            if timestamp:
                start_time, end_time = timestamp
                youtube_link = f"{self.YOUTUBE_BASE_URL}&t={int(start_time)}"
                
                quote = Quote(
                    text=quote_text,
                    speaker=quote_data.get('speaker'),
                    topic=quote_data.get('topic', 'general'),
                    importance=quote_data.get('importance', 'medium'),
                    start_time=start_time,
                    end_time=end_time,
                    youtube_link=youtube_link,
                    file_source=transcript_file.name,
                    context=quote_data.get('context')
                )
                extracted_quotes.append(quote)
        
        return extracted_quotes
    
    def pattern_based_extraction(self, text: str) -> List[Dict]:
        """Fallback pattern-based quote extraction"""
        quotes = []
        
        # Patterns that often indicate important statements
        patterns = [
            (r"(?:The key (?:insight|idea|concept) is)[^.]+\.", "key_insight"),
            (r"(?:We realized that)[^.]+\.", "realization"),
            (r"(?:This is (?:a |))?(\d+x|\d+%)[^.]+\.", "metric"),
            (r"(?:The (?:main |)challenge is)[^.]+\.", "challenge"),
            (r"(?:What's (?:really |)amazing is)[^.]+\.", "breakthrough"),
            (r"(?:We can now)[^.]+\.", "capability"),
            (r"(?:This means)[^.]+\.", "implication"),
            (r"(?:The future (?:of |is))[^.]+\.", "future"),
        ]
        
        for pattern, topic in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:  # Max 2 per pattern
                if len(match) > 30:  # Minimum length
                    quotes.append({
                        'text': match,
                        'topic': topic,
                        'importance': 'medium',
                        'speaker': None,
                        'context': None
                    })
        
        return quotes
    
    def extract_all_quotes(self) -> List[Quote]:
        """Extract quotes from all transcript files"""
        transcript_dir = Path("data/youtube_analysis/transcripts")
        all_quotes = []
        
        transcript_files = sorted(transcript_dir.glob("*_transcript.json"))
        transcript_files = [f for f in transcript_files if "claude" not in f.name]
        
        for transcript_file in transcript_files:
            print(f"Extracting from {transcript_file.name}...")
            quotes = self.extract_key_quotes(transcript_file)
            all_quotes.extend(quotes)
            print(f"  Found {len(quotes)} quotes")
        
        # Sort by importance and time
        all_quotes.sort(key=lambda q: (
            {'critical': 0, 'high': 1, 'medium': 2}.get(q.importance, 3),
            q.start_time
        ))
        
        self.quotes = all_quotes
        return all_quotes
    
    def generate_quote_report(self) -> str:
        """Generate a markdown report of key quotes with links"""
        report = []
        report.append("# Key Quotes from Bitcoin++ Istanbul Conference\n")
        report.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        report.append("Click timestamps to jump directly to the quote in the video.\n\n")
        
        # Group by importance
        critical = [q for q in self.quotes if q.importance == 'critical']
        high = [q for q in self.quotes if q.importance == 'high']
        medium = [q for q in self.quotes if q.importance == 'medium']
        
        if critical:
            report.append("## 🔴 Critical Insights\n\n")
            for quote in critical:
                report.append(f"### [{quote.formatted_time}]({quote.youtube_link}) - {quote.topic}\n")
                if quote.speaker:
                    report.append(f"*Speaker: {quote.speaker}*\n\n")
                report.append(f"> {quote.text}\n\n")
                if quote.context:
                    report.append(f"**Context:** {quote.context}\n\n")
                report.append("---\n\n")
        
        if high:
            report.append("## 🟡 Key Technical Points\n\n")
            for quote in high[:10]:  # Limit to top 10
                report.append(f"### [{quote.formatted_time}]({quote.youtube_link}) - {quote.topic}\n")
                if quote.speaker:
                    report.append(f"*Speaker: {quote.speaker}*\n\n")
                report.append(f"> {quote.text}\n\n")
                report.append("---\n\n")
        
        if medium:
            report.append("## 📝 Notable Mentions\n\n")
            for quote in medium[:5]:  # Just top 5
                report.append(f"**[{quote.formatted_time}]({quote.youtube_link})** - {quote.topic}: ")
                report.append(f'"{quote.text[:100]}..."\n\n')
        
        # Add summary stats
        report.append(f"\n## Summary\n")
        report.append(f"- Total quotes extracted: {len(self.quotes)}\n")
        report.append(f"- Critical insights: {len(critical)}\n")
        report.append(f"- High importance: {len(high)}\n")
        report.append(f"- Video: [{self.YOUTUBE_BASE_URL}]({self.YOUTUBE_BASE_URL})\n")
        
        return "".join(report)
    
    def export_quotes_json(self, output_file: Path):
        """Export quotes as JSON with all metadata"""
        quotes_data = []
        for quote in self.quotes:
            quotes_data.append({
                "text": quote.text,
                "speaker": quote.speaker,
                "topic": quote.topic,
                "importance": quote.importance,
                "youtube_link": quote.youtube_link,
                "timestamp": quote.formatted_time,
                "start_seconds": quote.start_time,
                "end_seconds": quote.end_time,
                "source_file": quote.file_source,
                "context": quote.context
            })
        
        with open(output_file, 'w') as f:
            json.dump({
                "video_url": self.YOUTUBE_BASE_URL,
                "total_quotes": len(quotes_data),
                "extraction_date": datetime.now().isoformat(),
                "quotes": quotes_data
            }, f, indent=2)
        
        print(f"Exported {len(quotes_data)} quotes to {output_file}")