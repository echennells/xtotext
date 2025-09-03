"""
Mention Detector - Finds mentions of specific people/topics in transcripts
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from datetime import datetime


class MentionDetector:
    """Detects mentions of specific people/topics in transcripts"""
    
    def __init__(self, search_terms: Optional[List[str]] = None):
        """
        Initialize the mention detector
        
        Args:
            search_terms: List of terms to search for (case-insensitive)
        """
        self.search_terms = search_terms or [
            "eric chennells",
            "lightning eric", 
            "podcast predictions",
            "prediction tracker",
            "predictiontracker"
        ]
        
        # Compile regex patterns for efficient searching
        self.patterns = []
        for term in self.search_terms:
            # Create pattern that matches the term with word boundaries
            # This prevents matching "eric" in "generic" for example
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            self.patterns.append((term, pattern))
    
    def detect_mentions(self, transcript_path: Path) -> Dict[str, Any]:
        """
        Detect mentions in a transcript file
        
        Args:
            transcript_path: Path to the transcript JSON file
            
        Returns:
            Dictionary with mention details
        """
        # Load transcript
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
        
        text = transcript_data.get('text', '')
        segments = transcript_data.get('segments', [])
        
        # Find all mentions
        mentions = []
        
        # Search in full text first to get all mentions
        for search_term, pattern in self.patterns:
            matches = list(pattern.finditer(text))
            
            for match in matches:
                mention = {
                    'term': search_term,
                    'position': match.start(),
                    'context': self._get_context(text, match.start(), match.end()),
                    'timestamp': self._find_timestamp(segments, match.start(), text)
                }
                mentions.append(mention)
        
        # Sort mentions by position
        mentions.sort(key=lambda x: x['position'])
        
        # Create summary
        summary = {
            'transcript_file': transcript_path.name,
            'total_mentions': len(mentions),
            'mentions_by_term': self._count_by_term(mentions),
            'mentions': mentions,
            'has_mentions': len(mentions) > 0,
            'detection_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _get_context(self, text: str, start: int, end: int, context_chars: int = 200) -> str:
        """Get surrounding context for a mention"""
        # Get context before and after the mention
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        
        # Find sentence boundaries
        # Look for previous sentence end
        prev_period = text.rfind('. ', context_start, start)
        if prev_period != -1:
            context_start = prev_period + 2
        
        # Look for next sentence end
        next_period = text.find('. ', end, context_end)
        if next_period != -1:
            context_end = next_period + 1
        
        # Extract and clean context
        before = text[context_start:start]
        mention = text[start:end]
        after = text[end:context_end]
        
        # Clean up whitespace
        before = ' '.join(before.split())
        after = ' '.join(after.split())
        
        return f"...{before} **{mention}** {after}..."
    
    def _find_timestamp(self, segments: List[Dict], position: int, full_text: str) -> Optional[float]:
        """Find the timestamp for a given text position"""
        if not segments:
            return None
        
        # Calculate which segment this position falls into
        current_pos = 0
        for segment in segments:
            segment_text = segment.get('text', '')
            segment_length = len(segment_text)
            
            if current_pos <= position < current_pos + segment_length:
                return segment.get('start', 0.0)
            
            current_pos += segment_length + 1  # +1 for space between segments
        
        return None
    
    def _count_by_term(self, mentions: List[Dict]) -> Dict[str, int]:
        """Count mentions by search term"""
        counts = {}
        for mention in mentions:
            term = mention['term']
            counts[term] = counts.get(term, 0) + 1
        return counts
    
    def process_batch(self, transcript_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Process all transcripts in a directory
        
        Args:
            transcript_dir: Directory containing transcript JSON files
            output_dir: Directory to save mention reports
            
        Returns:
            Summary of all mentions found
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all transcript files
        transcript_files = list(transcript_dir.glob("*_transcript.json"))
        
        all_mentions = []
        files_with_mentions = []
        
        for transcript_file in transcript_files:
            print(f"Checking {transcript_file.name} for mentions...")
            
            # Detect mentions
            result = self.detect_mentions(transcript_file)
            
            if result['has_mentions']:
                files_with_mentions.append(transcript_file.name)
                all_mentions.extend(result['mentions'])
                
                # Save individual report
                report_file = output_dir / f"{transcript_file.stem}_mentions.json"
                with open(report_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"  Found {result['total_mentions']} mentions!")
                for term, count in result['mentions_by_term'].items():
                    print(f"    - {term}: {count}")
        
        # Create overall summary
        summary = {
            'total_files_processed': len(transcript_files),
            'files_with_mentions': len(files_with_mentions),
            'total_mentions': len(all_mentions),
            'mentions_by_term': self._count_by_term(all_mentions),
            'files_with_mentions_list': files_with_mentions,
            'search_terms': self.search_terms,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = output_dir / "mention_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def generate_report(self, summary: Dict[str, Any], output_path: Path):
        """Generate a human-readable report of mentions"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MENTION DETECTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {summary['processing_timestamp']}")
        report_lines.append(f"\nSearch terms: {', '.join(summary['search_terms'])}")
        report_lines.append(f"\nFiles processed: {summary['total_files_processed']}")
        report_lines.append(f"Files with mentions: {summary['files_with_mentions']}")
        report_lines.append(f"Total mentions found: {summary['total_mentions']}")
        
        if summary['mentions_by_term']:
            report_lines.append("\nMentions by term:")
            for term, count in sorted(summary['mentions_by_term'].items()):
                report_lines.append(f"  - {term}: {count}")
        
        if summary['files_with_mentions_list']:
            report_lines.append("\nFiles containing mentions:")
            for filename in summary['files_with_mentions_list']:
                report_lines.append(f"  - {filename}")
        
        report_lines.append("\n" + "=" * 80)
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also print to console
        print('\n'.join(report_lines))


if __name__ == "__main__":
    # Example usage
    detector = MentionDetector()
    
    # Process a single transcript
    transcript_path = Path("data/episodes/bitcoin_dive_bar_analysis/transcripts/example_transcript.json")
    if transcript_path.exists():
        result = detector.detect_mentions(transcript_path)
        print(f"Found {result['total_mentions']} mentions")
        
        for mention in result['mentions']:
            print(f"\n{mention['term']} at timestamp {mention['timestamp']}:")
            print(f"  {mention['context']}")