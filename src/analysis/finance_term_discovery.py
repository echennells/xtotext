#!/usr/bin/env python3
"""
Finance Term Discovery Module
Scans transcripts to find potential financial terms that might be slang, 
misspellings, or informal references to stocks/crypto
"""
from typing import List, Dict, Set, Tuple
import json
import re
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm.llm_client import OpenAIClient


class FinanceTermDiscovery:
    """Discover potential financial terms in transcripts"""
    
    def __init__(self):
        self.client = OpenAIClient()
        self.client.model = "gpt-4-turbo"  # Use a good model for analysis
        
        # Known mappings to help identify patterns
        self.known_mappings = {
            'misty': 'MSTR',
            'similar': 'SMLR',
            'samler': 'SMLR',
            'semler': 'SMLR',
            'btc': 'BTC',
            'bitcoin': 'BTC',
            'eth': 'ETH',
            'ethereum': 'ETH',
            'microstrategy': 'MSTR',
            'micro strategy': 'MSTR',
        }
        
        # Common financial context words
        self.context_words = {
            'price', 'dollar', 'bucks', 'shares', 'stock', 'crypto',
            'pump', 'dump', 'moon', 'buy', 'sell', 'hold', 'trade',
            'market', 'cap', 'yield', 'dividend', 'treasury', 'company',
            'going to', 'headed to', 'target', 'support', 'resistance'
        }
    
    def extract_unknown_terms(self, text: str, context_window: int = 50) -> List[Dict]:
        """
        Use LLM to find potential financial terms that aren't standard
        """
        # Split into manageable chunks
        chunk_size = 30000
        all_terms = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            
            prompt = """Analyze this podcast transcript and find ALL potential financial terms that might be:
1. Slang or nicknames for stocks/crypto (like "misty" for MSTR)
2. Misspellings from speech-to-text (like "similar" for Semler/SMLR)
3. Informal references to companies or assets
4. Any term that sounds financial but isn't a standard ticker

For each term found, provide:
- The exact term as it appears
- What you think it might actually be
- Context snippet showing how it's used
- Confidence (high/medium/low)

Focus on terms that appear near price discussions, predictions, or financial context.

Output as JSON array:
[
  {
    "term": "misty",
    "likely_meaning": "MSTR (MicroStrategy)",
    "context": "misty is going to 500",
    "confidence": "high"
  }
]

Transcript chunk:
""" + chunk

            try:
                response = self.client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a financial transcript analyzer specializing in identifying informal financial terminology."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                content = response['choices'][0]['message']['content']
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                
                terms = json.loads(content.strip())
                all_terms.extend(terms)
                
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
        
        return all_terms
    
    def find_repeated_unknowns(self, text: str) -> Dict[str, int]:
        """
        Find frequently repeated words that might be financial terms
        """
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Filter for potential financial terms
        potential_terms = {}
        
        for word, count in word_counts.items():
            # Skip if too short or too common
            if len(word) < 3 or count < 3:
                continue
                
            # Skip known common words
            if word in {'the', 'and', 'for', 'that', 'this', 'with', 'but', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'could', 'you', 'your', 'they', 'them', 'their', 'what', 'when', 'where', 'which', 'who', 'why', 'how'}:
                continue
            
            # Check if it appears near financial context
            if self._appears_in_financial_context(word, text):
                potential_terms[word] = count
        
        return dict(sorted(potential_terms.items(), key=lambda x: x[1], reverse=True)[:50])
    
    def _appears_in_financial_context(self, word: str, text: str) -> bool:
        """Check if word appears near financial context"""
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = list(re.finditer(pattern, text.lower()))
        
        if not matches:
            return False
        
        # Check context around first few occurrences
        for match in matches[:5]:
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].lower()
            
            # Check for financial context words
            if any(ctx_word in context for ctx_word in self.context_words):
                return True
        
        return False
    
    def analyze_transcript(self, transcript_path: str) -> Dict:
        """
        Full analysis of a transcript for financial terms
        """
        print(f"\nAnalyzing transcript: {transcript_path}")
        
        with open(transcript_path, 'r') as f:
            data = json.load(f)
            text = data.get('text', '')
        
        print("Finding unknown financial terms with LLM...")
        unknown_terms = self.extract_unknown_terms(text)
        
        print("Finding frequently repeated potential terms...")
        repeated_terms = self.find_repeated_unknowns(text)
        
        # Combine and deduplicate
        term_analysis = {
            'transcript': Path(transcript_path).name,
            'unknown_terms': unknown_terms,
            'frequent_terms': repeated_terms,
            'summary': self._summarize_findings(unknown_terms, repeated_terms)
        }
        
        return term_analysis
    
    def _summarize_findings(self, unknown_terms: List[Dict], repeated_terms: Dict[str, int]) -> Dict:
        """Summarize the findings"""
        # Group unknown terms by likely meaning
        mappings = {}
        for term in unknown_terms:
            likely = term.get('likely_meaning', 'Unknown')
            if likely not in mappings:
                mappings[likely] = []
            mappings[likely].append({
                'term': term['term'],
                'confidence': term['confidence'],
                'context': term.get('context', '')
            })
        
        return {
            'total_unknown_terms': len(unknown_terms),
            'high_confidence_terms': len([t for t in unknown_terms if t.get('confidence') == 'high']),
            'potential_mappings': mappings,
            'top_repeated_unknowns': dict(list(repeated_terms.items())[:10])
        }
    
    def analyze_all_transcripts(self, transcript_dir: str) -> List[Dict]:
        """Analyze all transcripts in a directory"""
        results = []
        transcript_files = list(Path(transcript_dir).glob("*.json"))
        
        print(f"Found {len(transcript_files)} transcripts to analyze")
        
        for i, transcript_file in enumerate(transcript_files):
            print(f"\n[{i+1}/{len(transcript_files)}] Processing {transcript_file.name}...")
            
            try:
                result = self.analyze_transcript(str(transcript_file))
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {transcript_file}: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save analysis results"""
        output = {
            'analysis_date': str(Path(__file__).stat().st_mtime),
            'total_transcripts': len(results),
            'results': results,
            'discovered_mappings': self._consolidate_mappings(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def _consolidate_mappings(self, results: List[Dict]) -> Dict:
        """Consolidate all discovered mappings across transcripts"""
        consolidated = {}
        
        for result in results:
            summary = result.get('summary', {})
            mappings = summary.get('potential_mappings', {})
            
            for ticker, variants in mappings.items():
                if ticker not in consolidated:
                    consolidated[ticker] = set()
                
                for variant in variants:
                    if variant.get('confidence') in ['high', 'medium']:
                        consolidated[ticker].add(variant['term'])
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in consolidated.items()}