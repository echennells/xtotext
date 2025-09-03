"""
Post-process transcripts using Claude 3 Haiku via OpenRouter
More aggressive cleaning than GPT-3.5 version
"""
import os
import json
import time
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import tiktoken
from utils.filename_utils import sanitize_filename


class ClaudeTranscriptPostProcessor:
    """Post-process transcripts using Claude 3 Haiku for better cleaning"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-sonnet-4",
        max_chunk_tokens: int = 3000,  # Keep chunks small so output (4096 tokens) can contain the full corrected text
        temperature: float = 0.3
    ):
        """
        Initialize the Claude post-processor
        
        Args:
            api_key: OpenRouter API key (defaults to env var)
            model: Model to use (default: claude-3-haiku)
            max_chunk_tokens: Max tokens per chunk
            temperature: Temperature for corrections
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        # If not in env, try to load from config
        if not self.api_key:
            try:
                from config.config import OPENROUTER_API_KEY
                self.api_key = OPENROUTER_API_KEY
            except ImportError:
                pass
        
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or add to config/config.py")
        
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens
        self.temperature = temperature
        self.api_base = "https://openrouter.ai/api/v1"
        
        # Simple prompt for fixing transcription errors only
        self.system_prompt = """Fix any words or phrases that appear to be transcribed incorrectly based on context. Do not remove any content. Only fix obvious transcription mistakes.

OUTPUT: Return ONLY the corrected transcript."""
    
    def count_tokens(self, text: str) -> int:
        """Estimate tokens for Claude (rough approximation)"""
        # Claude tokenization is different but roughly 1 token = 4 chars
        return len(text) // 4
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        # Split by paragraphs first, then sentences
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            # No paragraphs, split by sentences
            sentences = text.replace('. ', '.\n').split('\n')
        else:
            sentences = []
            for para in paragraphs:
                sentences.extend(para.split('. '))
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back if missing
            if sentence and not sentence.endswith('.'):
                sentence += '.'
                
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_chunk_tokens and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_chunk(self, chunk: str, context: Optional[str] = None) -> str:
        """Process a single chunk with Claude"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/xtotext",
            "X-Title": "BSV Transcript Processor"
        }
        
        messages = []
        
        if context:
            user_content = f"Previous context for continuity:\n{context[-1000:]}\n\n---\n\nFix transcription errors in this text. Return ONLY the corrected text itself, with no preamble, no explanation, no 'Here is...' - just the cleaned transcript text:\n{chunk}"
        else:
            user_content = f"Fix transcription errors in this text. Return ONLY the corrected text itself, with no preamble, no explanation, no 'Here is...' - just the cleaned transcript text:\n{chunk}"
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 4096,  # Claude 3 Haiku's max output limit
            "system": self.system_prompt
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                print(f"Unexpected response format: {result}")
                return chunk
                
        except Exception as e:
            print(f"Error processing chunk with Claude: {e}")
            return chunk
    
    def process_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """Process a full transcript"""
        print("Post-processing transcript with Claude 3 Haiku...")
        
        # Extract full text
        if 'text' in transcript:
            full_text = transcript['text']
        else:
            full_text = ' '.join(seg.get('text', '') for seg in transcript.get('segments', []))
        
        # Split into chunks
        chunks = self.split_into_chunks(full_text)
        print(f"Processing {len(chunks)} chunks...")
        
        # Process each chunk
        corrected_chunks = []
        context = ""
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            corrected = self.process_chunk(chunk, context)
            
            # Clean up any preambles that slipped through
            for preamble in ["Here is the cleaned", "Here's the cleaned", "Sure, here", "I'll clean"]:
                if corrected.startswith(preamble):
                    # Find the first real content (usually after a colon or newline)
                    for sep in [':', '\n\n', '\n']:
                        if sep in corrected:
                            corrected = corrected.split(sep, 1)[1].strip()
                            break
            
            corrected_chunks.append(corrected)
            context = corrected  # Use corrected text as context
            
            # Rate limiting for OpenRouter
            if i < len(chunks) - 1:
                time.sleep(0.5)
        
        # Combine corrected text with double newlines between chunks
        corrected_text = '\n\n'.join(corrected_chunks)
        
        # Update transcript
        result = transcript.copy()
        result['text'] = corrected_text
        result['postprocessed'] = True
        result['postprocess_model'] = self.model
        
        return result
    
    def process_file(self, input_path: Path, output_path: Optional[Path] = None, save_text_files: bool = True) -> Path:
        """Process a transcript file"""
        # Load transcript
        with open(input_path, 'r') as f:
            transcript = json.load(f)
        
        # Extract original text
        original_text = transcript.get('text', '')
        if not original_text and 'segments' in transcript:
            original_text = ' '.join(seg.get('text', '') for seg in transcript['segments'])
        
        # Process
        corrected = self.process_transcript(transcript)
        
        # Save JSON with sanitized filename
        if not output_path:
            sanitized_name = sanitize_filename(f"{input_path.stem}_claude_postprocessed.json")
            output_path = input_path.parent / sanitized_name
        
        with open(output_path, 'w') as f:
            json.dump(corrected, f, indent=2)
        
        # Save text files for comparison
        if save_text_files:
            # Save before text with sanitized filename
            before_name = sanitize_filename(f"{input_path.stem}_claude_before.txt")
            before_path = input_path.parent / before_name
            with open(before_path, 'w') as f:
                f.write(original_text)
            print(f"Saved original text to: {before_path}")
            
            # Save after text with sanitized filename
            after_name = sanitize_filename(f"{input_path.stem}_claude_after.txt")
            after_path = input_path.parent / after_name
            with open(after_path, 'w') as f:
                f.write(corrected['text'])
            print(f"Saved corrected text to: {after_path}")
        
        print(f"Saved post-processed transcript to: {output_path}")
        return output_path


# Convenience function
def postprocess_transcript_claude(
    transcript_path: Path,
    output_path: Optional[Path] = None,
    api_key: Optional[str] = None
) -> Path:
    """
    Post-process a transcript with Claude 3 Haiku
    
    Args:
        transcript_path: Path to transcript JSON
        output_path: Output path (optional)
        api_key: OpenRouter API key (optional)
        
    Returns:
        Path to processed file
    """
    processor = ClaudeTranscriptPostProcessor(api_key=api_key)
    return processor.process_file(transcript_path, output_path)