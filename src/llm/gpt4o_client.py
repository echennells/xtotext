"""
GPT-4o client - Balanced model for focused extraction
Best for: Extracting structured predictions from identified snippets
"""
from typing import List, Dict, Optional, Any
from openai import OpenAI
from datetime import datetime
from .base_client import BaseLLMClient, ExtractionStage


class GPT4OClient(BaseLLMClient):
    """
    GPT-4o: Good balance of cost and capability
    - $2.50/1M input tokens
    - $10.00/1M output tokens
    - 128k context window
    - Supports JSON mode
    - Better reasoning than mini
    """
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
    
    @property
    def model_name(self) -> str:
        return "gpt-4o"
    
    @property
    def supports_json_mode(self) -> bool:
        return True
    
    @property
    def supports_temperature(self) -> bool:
        return True
    
    @property
    def context_window(self) -> int:
        return 128000  # 128k tokens
    
    @property
    def chunk_size(self) -> int:
        # Moderate chunks for focused work
        # 50k chars ≈ 12.5k tokens
        return 50000
    
    @property
    def cost_per_million_input(self) -> float:
        return 2.50  # $2.50 per million input tokens
    
    @property
    def cost_per_million_output(self) -> float:
        return 10.00  # $10.00 per million output tokens
    
    def prepare_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Standard message format for GPT models"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call OpenAI API"""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.0,
            "max_tokens": max_tokens if max_tokens is not None else 4000,
        }
        
        # Add response format if specified
        if response_format:
            kwargs["response_format"] = response_format
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            
            # Convert to expected format
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def extract_predictions(self, snippets: List[Dict], episode_info: Dict) -> List[Dict]:
        """
        Extract structured predictions from snippets
        This is the main value-add of GPT-4o over mini
        """
        system_prompt = """You are extracting cryptocurrency price predictions from podcast snippets.

For each snippet, identify:
1. The speaker making the prediction (if identifiable)
2. The asset being predicted (BTC, ETH, MSTR, etc.)
3. The target price or percentage
4. The timeframe - COPY EXACTLY what the speaker said about timing (e.g. "next week", "by end of year", "when we break 100k")
5. Confidence level (based on how they phrase it)
6. Any conditions mentioned

Be precise and only extract predictions made BY THE SPEAKERS, not discussions about others' predictions.

ASSET MAPPING:
- "misty" → MSTY
- "similar", "silver", "sampler", "summer" → SMLR  
- "mst" → MSTR
- "big coin" → BTC
- "meta planet" → METAPLANET
- "I bet", "IBIT" → IBIT
- "galaxy" → GLXY

IMPORTANT: For timeframe, just copy the exact words used. Do NOT parse, categorize, or interpret. Examples:
- If they say "next Tuesday" → timeframe: "next Tuesday"
- If they say "by the end of this cycle" → timeframe: "by the end of this cycle"
- If they say "tomorrow" → timeframe: "tomorrow"
- If no timeframe mentioned → timeframe: ""

Output JSON:
{
  "predictions": [
    {
      "speaker": "Speaker name or Unknown",
      "asset": "BTC",
      "price": 150000,
      "timeframe": "end of 2025",
      "confidence": "high/medium/low",
      "conditions": "if ETF approval happens",
      "reasoning": "their stated logic",
      "quote": "exact quote from transcript showing the prediction"
    }
  ]
}"""

        all_predictions = []
        
        # Process snippets in batches to save on API calls
        batch_size = 5
        for i in range(0, len(snippets), batch_size):
            batch = snippets[i:i+batch_size]
            
            # Prepare batch text
            batch_text = "\n\n---SNIPPET BREAK---\n\n".join([
                f"[Snippet {j}]\n{snippet['text']}" 
                for j, snippet in enumerate(batch, i)
            ])
            
            response = self.extract_for_stage(
                stage=ExtractionStage.FOCUSED_SCAN,
                text=batch_text,
                system_prompt=system_prompt,
                temperature=0.0,
                parse_json=True
            )
            
            if response and 'predictions' in response:
                # Add episode info to each prediction
                for pred in response['predictions']:
                    pred['episode'] = episode_info.get('title', '')
                    pred['episode_date'] = episode_info.get('date', '')
                    pred['extraction_date'] = datetime.now().isoformat()
                    all_predictions.append(pred)
        
        return all_predictions
    
    def refine_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """
        Refine and normalize predictions
        Fix common issues like asset names, validate prices, etc.
        """
        system_prompt = """You are refining cryptocurrency predictions for consistency.

Tasks:
1. Normalize asset names (e.g., "Bitcoin" -> "BTC", "MicroStrategy" -> "MSTR")
2. Validate price targets (remove obvious errors)
3. Standardize timeframes (e.g., "EOY" -> "end of 2024")
4. Merge duplicate predictions
5. Flag suspicious predictions

Input will be a list of predictions. Output the refined list.

Common asset mappings:
- Bitcoin, BTC, bitcoin -> BTC
- Ethereum, ETH, ether -> ETH
- MicroStrategy, MSTR, Saylor's company -> MSTR
- Solana, SOL -> SOL
- Semler, SMLR -> SMLR

Output JSON:
{
  "refined_predictions": [...],
  "removed_predictions": [
    {"prediction": {...}, "reason": "why it was removed"}
  ]
}"""

        if not predictions:
            return []
        
        # Process in chunks if many predictions
        chunk_size = 20
        all_refined = []
        
        for i in range(0, len(predictions), chunk_size):
            chunk = predictions[i:i+chunk_size]
            
            response = self.extract_for_stage(
                stage=ExtractionStage.FOCUSED_SCAN,
                text=json.dumps(chunk, indent=2),
                system_prompt=system_prompt,
                temperature=0.0,
                parse_json=True,
                max_tokens=6000  # Need more tokens for refinement
            )
            
            if response and 'refined_predictions' in response:
                all_refined.extend(response['refined_predictions'])
        
        return all_refined