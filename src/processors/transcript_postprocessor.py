"""
Post-process transcripts to fix common transcription errors using LLM
Focuses on crypto/finance terminology, proper nouns, and contextual corrections
"""
import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import tiktoken
from openai import OpenAI


class TranscriptPostProcessor:
    """Post-process transcripts to fix transcription errors"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_chunk_tokens: int = 3000,
        temperature: float = 0.3
    ):
        """
        Initialize the post-processor
        
        Args:
            api_key: OpenAI API key (defaults to env var)
            model: Model to use (gpt-3.5-turbo or gpt-4)
            max_chunk_tokens: Max tokens per chunk
            temperature: Temperature for corrections (lower = more conservative)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # If not in env, try to load from config
        if not self.api_key:
            try:
                from config.config import OPENAI_API_KEY
                self.api_key = OPENAI_API_KEY
            except ImportError:
                pass
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or add to config/config.py")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens
        self.temperature = temperature
        self.encoding = tiktoken.encoding_for_model(model)
        
        # System prompt for crypto/finance transcription fixing
        self.system_prompt = """You are a transcript editor specializing in cryptocurrency and finance discussions. 
Your task is to fix obvious transcription errors while preserving the original meaning and speaker's voice.

Focus on:
1. Crypto/finance terms: Bitcoin, Ethereum, stablecoins, DeFi, etc.
2. Proper nouns: Company names, people's names, project names
3. Technical terms that sound similar but have different meanings
4. Number/price corrections when obviously wrong
5. Acronyms and their proper capitalization (BTC, ETH, USD, etc.)

Rules:
- Only fix obvious errors, don't rewrite or paraphrase
- Preserve the conversational nature and filler words
- Keep timestamps and speaker labels intact
- If unsure, leave it unchanged
- Common corrections: "beat coin" → "Bitcoin", "ether room" → "Ethereum", "stable coins" → "stablecoins"
- IMPORTANT: Return ONLY the corrected transcript text. Do not add any preambles like "Here is the corrected transcript" or "Sure, here is...". Just return the corrected text directly.
"""
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks that fit within token limits"""
        # Split by sentences first
        sentences = text.replace('\n', ' \n ').split('. ')
        sentences = [s.strip() + '.' if not s.endswith('.') else s for s in sentences]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
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
        """Process a single chunk of transcript"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if context:
            messages.append({
                "role": "user", 
                "content": f"Previous context for continuity:\n{context[-500:]}\n\n---\n\nFix transcription errors in:\n{chunk}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Fix transcription errors in this transcript:\n{chunk}"
            })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_chunk_tokens + 500  # Allow some expansion
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return chunk  # Return original if error
    
    def process_transcript(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a full transcript JSON
        
        Args:
            transcript: Transcript dict with 'text' and 'segments'
            
        Returns:
            Updated transcript with corrections
        """
        print("Post-processing transcript for error correction...")
        
        # Extract full text
        if 'text' in transcript:
            full_text = transcript['text']
        else:
            # Build from segments
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
            corrected_chunks.append(corrected)
            context = corrected  # Use corrected text as context for next chunk
            
            # Rate limiting
            if i < len(chunks) - 1:
                time.sleep(1)  # Avoid rate limits
        
        # Combine corrected text
        corrected_text = ' '.join(corrected_chunks)
        
        # Update transcript
        result = transcript.copy()
        result['text'] = corrected_text
        result['postprocessed'] = True
        result['postprocess_model'] = self.model
        
        # Also update segments if they exist
        if 'segments' in result and result['segments']:
            # Simple approach: just mark as postprocessed
            # More sophisticated approach would map corrections back to segments
            result['segments_need_update'] = True
        
        return result
    
    def process_file(self, input_path: Path, output_path: Optional[Path] = None, save_text_files: bool = True) -> Path:
        """
        Process a transcript file
        
        Args:
            input_path: Path to input transcript JSON
            output_path: Path for output (defaults to input_postprocessed.json)
            save_text_files: Save before/after .txt files for comparison
            
        Returns:
            Path to output file
        """
        # Load transcript
        with open(input_path, 'r') as f:
            transcript = json.load(f)
        
        # Extract original text
        original_text = transcript.get('text', '')
        if not original_text and 'segments' in transcript:
            original_text = ' '.join(seg.get('text', '') for seg in transcript['segments'])
        
        # Process
        corrected = self.process_transcript(transcript)
        
        # Clean up any GPT preambles
        corrected_text = corrected['text']
        if corrected_text.startswith('Sure, here is the revised transcript:'):
            corrected_text = corrected_text[37:].strip()
            corrected['text'] = corrected_text
        elif corrected_text.startswith('Here is the corrected transcript:'):
            corrected_text = corrected_text[33:].strip()
            corrected['text'] = corrected_text
        
        # Save JSON
        if not output_path:
            output_path = input_path.parent / f"{input_path.stem}_postprocessed.json"
        
        with open(output_path, 'w') as f:
            json.dump(corrected, f, indent=2)
        
        # Save text files for comparison
        if save_text_files:
            # Save before text
            before_path = input_path.parent / f"{input_path.stem}_before.txt"
            with open(before_path, 'w') as f:
                f.write(original_text)
            print(f"Saved original text to: {before_path}")
            
            # Save after text
            after_path = input_path.parent / f"{input_path.stem}_after.txt"
            with open(after_path, 'w') as f:
                f.write(corrected_text)
            print(f"Saved corrected text to: {after_path}")
        
        print(f"Saved post-processed transcript to: {output_path}")
        return output_path


# Convenience function for easy use
def postprocess_transcript(
    transcript_path: Path,
    output_path: Optional[Path] = None,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> Path:
    """
    Post-process a transcript file to fix errors
    
    Args:
        transcript_path: Path to transcript JSON
        output_path: Output path (optional)
        model: Model to use (gpt-3.5-turbo or gpt-4)
        api_key: OpenAI API key (optional, uses env var)
        
    Returns:
        Path to processed file
    """
    processor = TranscriptPostProcessor(api_key=api_key, model=model)
    return processor.process_file(transcript_path, output_path)