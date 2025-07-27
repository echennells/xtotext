"""
Custom prompt version of the two-stage extractor
Allows using prompts generated from Rogue Trader analysis
"""
from typing import List, Dict, Optional
import json
from pathlib import Path

from .llm_extractor_two_stage import TwoStageLLMExtractor
from .models import Prediction


class CustomPromptExtractor(TwoStageLLMExtractor):
    """Extended version that allows custom prompts"""
    
    def __init__(self, stage1_prompt: Optional[str] = None, stage2_prompt: Optional[str] = None):
        super().__init__()
        self.custom_stage1_prompt = stage1_prompt
        self.custom_stage2_prompt = stage2_prompt
        
    def extract_snippets(self, text: str, context: Dict, start_offset: int = 0) -> List[Dict]:
        """Override to use custom stage 1 prompt if provided"""
        
        if self.custom_stage1_prompt:
            system_prompt = self.custom_stage1_prompt
        else:
            # Use default prompt
            system_prompt = """You are a snippet extractor for a Bitcoin/crypto podcast. 
Your job is to find parts of the conversation that MIGHT contain price predictions made BY THE SPEAKERS THEMSELVES.

[... rest of default prompt ...]"""
        
        # Prepare the messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Find prediction snippets in this text:

{text}

Remember to include 30-45 seconds of context (150-200 words) around each potential prediction."""}
        ]
        
        # Call the LLM
        response = self.snippet_client.chat_completion(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # Track tokens
        self.token_usage['snippet_extraction']['calls'] += 1
        self.token_usage['snippet_extraction']['input'] += response.get('usage', {}).get('prompt_tokens', 0)
        self.token_usage['snippet_extraction']['output'] += response.get('usage', {}).get('completion_tokens', 0)
        
        # Parse response
        try:
            content = response['choices'][0]['message']['content']
            if isinstance(content, str):
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                result = json.loads(content.strip())
            else:
                result = content
                
            snippets = result.get('snippets', [])
            
            # Add context to each snippet
            for snippet in snippets:
                snippet['context'] = context
                snippet['char_start'] = snippet.get('char_start', 0) + start_offset
                snippet['char_end'] = snippet.get('char_end', 0) + start_offset
                
            return snippets
            
        except Exception as e:
            print(f"Error parsing snippet response: {e}")
            return []
    
    def extract_predictions_from_snippets(self, snippets: List[Dict], episode_info: Dict) -> List[Prediction]:
        """Override to use custom stage 2 prompt if provided"""
        
        if self.custom_stage2_prompt:
            base_prompt = self.custom_stage2_prompt
        else:
            # Use default prompt
            base_prompt = """Extract specific predictions from this snippet of a finance/crypto podcast.

[... rest of default prompt ...]"""
        
        predictions = []
        
        for i, snippet in enumerate(snippets):
            # Prepare the prompt
            system_prompt = base_prompt
            
            user_prompt = f"""Extract predictions from this snippet:

{snippet['text']}

Episode info:
- Title: {episode_info.get('title', 'Unknown')}
- Upload date: {episode_info.get('upload_date', 'Unknown')}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                # Call the prediction model
                response = self.prediction_client.chat_completion(
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                
                # Track tokens
                self.token_usage['prediction_extraction']['calls'] += 1
                self.token_usage['prediction_extraction']['input'] += response.get('usage', {}).get('prompt_tokens', 0)
                self.token_usage['prediction_extraction']['output'] += response.get('usage', {}).get('completion_tokens', 0)
                
                # Parse response
                content = response['choices'][0]['message']['content']
                if isinstance(content, str):
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    result = json.loads(content.strip())
                else:
                    result = content
                
                # Process predictions
                snippet_predictions = result.get('predictions', [])
                
                for pred_data in snippet_predictions:
                    # Create prediction object
                    prediction = self._create_prediction_from_data(pred_data, snippet, episode_info)
                    if prediction:
                        predictions.append(prediction)
                        
            except Exception as e:
                print(f"Error extracting predictions from snippet {i+1}: {e}")
                
        return predictions
    
    def _create_prediction_from_data(self, pred_data: Dict, snippet: Dict, episode_info: Dict) -> Optional[Prediction]:
        """Create a Prediction object from extracted data"""
        try:
            # Normalize asset
            asset = pred_data.get('asset', '').upper()
            if asset.lower() in self.asset_map:
                asset = self.asset_map[asset.lower()]
            
            prediction = Prediction(
                asset=asset,
                price=pred_data.get('price', ''),
                date=pred_data.get('date', ''),
                speaker=pred_data.get('speaker', 'Unknown'),
                context=pred_data.get('context', snippet.get('text', '')),
                confidence=pred_data.get('confidence', 'medium'),
                prediction_type=pred_data.get('type', 'price_target'),
                timeframe=pred_data.get('timeframe', 'medium_term'),
                condition=pred_data.get('condition'),
                rationale=pred_data.get('rationale'),
                snippet_text=snippet.get('text', ''),
                char_position=snippet.get('char_start', 0),
                episode_title=episode_info.get('title', ''),
                episode_date=episode_info.get('upload_date', ''),
                extracted_at=self._get_current_timestamp()
            )
            
            return prediction
            
        except Exception as e:
            print(f"Error creating prediction object: {e}")
            return None
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


def load_and_extract_with_custom_prompts(prompts_file: str, transcript_file: str, episode_info: Dict) -> List[Prediction]:
    """Helper function to load prompts and extract predictions"""
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # Create extractor with custom prompts
    extractor = CustomPromptExtractor(
        stage1_prompt=prompts.get('stage1'),
        stage2_prompt=prompts.get('stage2')
    )
    
    # Extract predictions
    return extractor.extract_predictions_from_file(transcript_file, episode_info)