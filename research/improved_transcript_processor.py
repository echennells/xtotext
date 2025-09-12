"""
Improved Transcript Processor with intelligent content extraction
Uses modern models with large context windows
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from openai import OpenAI
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.config import OPENROUTER_API_KEY


@dataclass 
class ProcessingStrategy:
    """Configuration for different processing strategies"""
    model: str
    context_window: int
    cost_per_million_input: float
    cost_per_million_output: float
    chunk_overlap: int = 1000  # Overlap between chunks in characters
    

class ImprovedTranscriptProcessor:
    """
    Smart transcript processor that uses full context windows efficiently
    """
    
    # Model configurations based on the report
    MODELS = {
        "deepseek-v3": ProcessingStrategy(
            model="deepseek/deepseek-v3-base",
            context_window=164000,  # 164K tokens ~ 650K characters
            cost_per_million_input=0.20,
            cost_per_million_output=0.80
        ),
        "mistral-small": ProcessingStrategy(
            model="mistralai/mistral-small-3.1",
            context_window=128000,  # 128K tokens ~ 500K characters
            cost_per_million_input=0.25,
            cost_per_million_output=1.0
        ),
        "sonar-research": ProcessingStrategy(
            model="perplexity/sonar-research",
            context_window=128000,
            cost_per_million_input=0.30,
            cost_per_million_output=1.20
        ),
        "claude-sonnet": ProcessingStrategy(
            model="anthropic/claude-sonnet-4",
            context_window=200000,  # Actually 1M tokens for Claude 4!
            cost_per_million_input=3.0,
            cost_per_million_output=15.0
        )
    }
    
    def __init__(self, api_key: Optional[str] = None, strategy: str = "deepseek-v3"):
        """Initialize with chosen strategy"""
        self.api_key = api_key or OPENROUTER_API_KEY
        self.strategy = self.MODELS.get(strategy, self.MODELS["deepseek-v3"])
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        self.total_tokens = {"input": 0, "output": 0}
    
    def extract_relevant_sections(self, text: str, keywords: List[str], 
                                 window_size: int = 2000) -> List[Dict[str, Any]]:
        """
        Extract relevant sections around keywords instead of just taking first 15K chars
        """
        relevant_sections = []
        text_lower = text.lower()
        
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword.lower()))
            for match in pattern.finditer(text_lower):
                start = max(0, match.start() - window_size)
                end = min(len(text), match.end() + window_size)
                
                # Find paragraph boundaries
                para_start = text.rfind('\n\n', start, match.start())
                if para_start == -1:
                    para_start = start
                    
                para_end = text.find('\n\n', match.end(), end)
                if para_end == -1:
                    para_end = end
                
                relevant_sections.append({
                    "keyword": keyword,
                    "position": match.start(),
                    "context": text[para_start:para_end].strip(),
                    "start": para_start,
                    "end": para_end
                })
        
        # Merge overlapping sections
        merged = self._merge_overlapping_sections(relevant_sections)
        return merged
    
    def _merge_overlapping_sections(self, sections: List[Dict]) -> List[Dict]:
        """Merge overlapping text sections to avoid duplication"""
        if not sections:
            return []
        
        # Sort by start position
        sections.sort(key=lambda x: x['start'])
        
        merged = []
        current = sections[0].copy()
        current['keywords'] = [current['keyword']]
        
        for section in sections[1:]:
            if section['start'] <= current['end']:
                # Overlapping - merge
                current['end'] = max(current['end'], section['end'])
                current['keywords'].append(section['keyword'])
            else:
                # Non-overlapping - save current and start new
                merged.append(current)
                current = section.copy()
                current['keywords'] = [section['keyword']]
        
        merged.append(current)
        return merged
    
    def analyze_with_full_context(self, transcripts: List[Dict[str, Any]], 
                                 analysis_prompt: str) -> Dict[str, Any]:
        """
        Use the full context window intelligently
        """
        # Calculate total characters available
        max_chars = self.strategy.context_window * 4  # Rough estimate: 1 token = 4 chars
        
        # Combine transcripts intelligently
        combined_text = ""
        metadata = []
        
        for transcript in transcripts:
            text = transcript.get('processed_text', transcript.get('text', ''))
            file_info = transcript.get('file_path', 'unknown')
            
            # Add with metadata
            section = f"\n\n=== TRANSCRIPT: {file_info} ===\n{text}\n"
            
            if len(combined_text) + len(section) < max_chars * 0.9:  # Leave 10% buffer
                combined_text += section
                metadata.append({
                    "file": file_info,
                    "words": len(text.split()),
                    "included": "full"
                })
            else:
                # Smart truncation - find relevant sections
                remaining_space = max_chars - len(combined_text) - 1000
                if remaining_space > 5000:
                    # Extract most relevant parts
                    keywords = ["bitvm", "garbled", "snark", "bridge", "operator", "challenge"]
                    relevant = self.extract_relevant_sections(text, keywords, window_size=1000)
                    
                    relevant_text = "\n...\n".join([s['context'] for s in relevant[:5]])
                    truncated_section = f"\n\n=== TRANSCRIPT (excerpts): {file_info} ===\n{relevant_text}\n"
                    
                    if len(truncated_section) < remaining_space:
                        combined_text += truncated_section
                        metadata.append({
                            "file": file_info,
                            "words": len(text.split()),
                            "included": "excerpts",
                            "num_excerpts": len(relevant[:5])
                        })
                break
        
        # Now process with full context
        try:
            response = self.client.chat.completions.create(
                model=self.strategy.model,
                messages=[
                    {"role": "system", "content": "You are analyzing technical conference transcripts. Provide detailed, structured analysis."},
                    {"role": "user", "content": f"{analysis_prompt}\n\nTranscripts:\n{combined_text}"}
                ],
                temperature=0.3,
                max_tokens=4096
            )
            
            result_text = response.choices[0].message.content
            
            # Update token tracking
            self.total_tokens["input"] += len(combined_text) // 4
            self.total_tokens["output"] += len(result_text) // 4
            
            # Try to parse as JSON if possible
            try:
                result = json.loads(result_text)
            except:
                result = {"analysis": result_text}
            
            return {
                "model": self.strategy.model,
                "metadata": metadata,
                "total_chars_analyzed": len(combined_text),
                "result": result,
                "cost": self.calculate_cost()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "metadata": metadata
            }
    
    def iterative_deep_analysis(self, transcripts: List[Dict[str, Any]], 
                               focus_topics: List[str]) -> Dict[str, Any]:
        """
        Perform iterative analysis: first extract key points, then synthesize
        """
        all_extracts = []
        
        # Phase 1: Extract from each transcript
        for transcript in transcripts:
            text = transcript.get('processed_text', transcript.get('text', ''))
            file_info = transcript.get('file_path', 'unknown')
            
            # Find relevant sections for each topic
            topic_sections = {}
            for topic in focus_topics:
                sections = self.extract_relevant_sections(
                    text, 
                    [topic], 
                    window_size=3000  # Larger window for more context
                )
                if sections:
                    topic_sections[topic] = sections
            
            if topic_sections:
                # Extract insights from relevant sections
                extract_prompt = f"""
                Extract key technical insights about these topics from the transcript:
                Topics: {', '.join(focus_topics)}
                
                For each topic found, provide:
                1. Key technical details
                2. Performance metrics or numbers mentioned
                3. Challenges or limitations discussed
                4. Future directions or improvements suggested
                
                Output as structured JSON.
                """
                
                # Combine relevant sections only
                relevant_text = ""
                for topic, sections in topic_sections.items():
                    relevant_text += f"\n\n--- {topic.upper()} SECTIONS ---\n"
                    for s in sections[:3]:  # Top 3 sections per topic
                        relevant_text += f"\n{s['context']}\n"
                
                if len(relevant_text) > 1000:  # Only analyze if substantial content
                    try:
                        response = self.client.chat.completions.create(
                            model=self.strategy.model,
                            messages=[
                                {"role": "system", "content": "Extract structured technical information from conference transcript excerpts."},
                                {"role": "user", "content": f"{extract_prompt}\n\nRelevant sections:\n{relevant_text[:50000]}"}
                            ],
                            temperature=0.2,
                            max_tokens=2048
                        )
                        
                        extract = {
                            "file": file_info,
                            "topics_found": list(topic_sections.keys()),
                            "insights": response.choices[0].message.content
                        }
                        all_extracts.append(extract)
                        
                    except Exception as e:
                        print(f"Error processing {file_info}: {e}")
        
        # Phase 2: Synthesize all extracts
        if all_extracts:
            synthesis_prompt = f"""
            Synthesize these technical insights from multiple conference talks into a comprehensive analysis.
            
            Create a structured report covering:
            1. Major Technical Innovations (with specific metrics)
            2. Common Challenges Across Talks
            3. Evolution of Ideas (how concepts build on each other)
            4. Key Technical Details and Performance Numbers
            5. Future Research Directions
            
            Focus on technical accuracy and specific details.
            """
            
            combined_extracts = json.dumps(all_extracts, indent=2)[:100000]  # Limit size
            
            try:
                response = self.client.chat.completions.create(
                    model=self.strategy.model,
                    messages=[
                        {"role": "system", "content": "You are synthesizing technical insights from multiple conference talks."},
                        {"role": "user", "content": f"{synthesis_prompt}\n\nExtracts:\n{combined_extracts}"}
                    ],
                    temperature=0.3,
                    max_tokens=4096
                )
                
                synthesis = response.choices[0].message.content
                
            except Exception as e:
                synthesis = f"Synthesis error: {e}"
        else:
            synthesis = "No relevant content found for specified topics"
        
        return {
            "phase1_extracts": all_extracts,
            "phase2_synthesis": synthesis,
            "topics_analyzed": focus_topics,
            "transcripts_processed": len(transcripts),
            "model_used": self.strategy.model,
            "total_cost": self.calculate_cost()
        }
    
    def calculate_cost(self) -> float:
        """Calculate processing cost"""
        input_cost = (self.total_tokens["input"] / 1_000_000) * self.strategy.cost_per_million_input
        output_cost = (self.total_tokens["output"] / 1_000_000) * self.strategy.cost_per_million_output
        return input_cost + output_cost
    
    def find_cross_references(self, transcripts: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Find where speakers reference each other or build on previous talks
        """
        references = defaultdict(list)
        
        # Common patterns for references
        reference_patterns = [
            r"as (\w+) (?:mentioned|said|discussed|showed)",
            r"(\w+)'s (?:talk|presentation|work|research)",
            r"building on (\w+)'s",
            r"(?:similar to|like) what (\w+)",
            r"(\w+) (?:already|just) (?:presented|explained|showed)"
        ]
        
        for i, transcript in enumerate(transcripts):
            text = transcript.get('processed_text', transcript.get('text', ''))
            file_info = transcript.get('file_path', 'unknown')
            
            for pattern in reference_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match) > 2:  # Filter out single letters
                        references[match.lower()].append({
                            "transcript": file_info,
                            "transcript_index": i,
                            "pattern": pattern,
                            "context": self._get_context(text, match, 200)
                        })
        
        return dict(references)
    
    def _get_context(self, text: str, term: str, window: int = 200) -> str:
        """Get context around a term"""
        idx = text.lower().find(term.lower())
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(term) + window)
        return "..." + text[start:end] + "..."