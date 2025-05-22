import os
import asyncio
import time
import json
import re
import logging
from typing import Tuple, Optional, List, Dict
from collections import defaultdict

import openai
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from llamaapi import LlamaAPI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

from config import MODEL_CONFIG, prompt_templates

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cost tracking
cost_tracker = defaultdict(int)

class ValidationError(Exception):
    pass

def validate_scale(scale_range: List[int]) -> List[int]:
    """Validate scale range is properly formatted"""
    if not (isinstance(scale_range, list) and len(scale_range) == 2 
            and isinstance(scale_range[0], int) and isinstance(scale_range[1], int)
            and scale_range[0] < scale_range[1]):
        raise ValidationError(f"Invalid scale range: {scale_range}")
    return scale_range

class SurveyAnswer(BaseModel):
    numeric_score: float
    label: str = ""  # Make optional with default
    justification: str = ""  # Make optional with default

def safe_parse_survey_answer(response_text: str, scale_range: List[int]) -> Optional[SurveyAnswer]:
    """
    Simplified parser that handles both structured and unstructured responses.
    Prioritizes finding valid numeric scores within the scale range.
    """
    response_text = str(response_text).strip().lower()  # Normalize text
    min_scale, max_scale = scale_range
    
    def is_valid_score(num: float) -> bool:
        return min_scale <= num <= max_scale

    try:
        # 1. Simple format: "Rating: X" or "Score: X"
        simple_match = re.search(r'(?:rating|score):\s*(-?\d+(?:\.\d+)?)', response_text)
        if simple_match:
            num = float(simple_match.group(1))
            if is_valid_score(num):
                # Get everything after the rating as justification
                explanation = response_text[simple_match.end():].strip()
                return SurveyAnswer(
                    numeric_score=num,
                    justification=explanation or response_text
                )

        # 2. Try JSON parsing (for OpenAI, Claude, Grok)
        try:
            data = json.loads(response_text)
            if 'rating' in data and is_valid_score(float(data['rating'])):
                return SurveyAnswer(
                    numeric_score=float(data['rating']),
                    justification=data.get('justification', '')
                )
        except json.JSONDecodeError:
            pass

        # 3. Find first valid number in text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response_text)
        for num_str in numbers:
            try:
                num = float(num_str)
                if is_valid_score(num):
                    return SurveyAnswer(
                        numeric_score=num,
                        justification=response_text
                    )
            except ValueError:
                continue

        # 4. Fallback: use scale midpoint
        midpoint = (min_scale + max_scale) / 2
        logger.warning(f"No valid number found in: {response_text[:100]}...")
        return SurveyAnswer(
            numeric_score=midpoint,
            justification=f"PARSER WARNING: No valid number found in response"
        )

    except Exception as e:
        logger.error(f"Parser error: {str(e)}")
        midpoint = (min_scale + max_scale) / 2
        return SurveyAnswer(
            numeric_score=midpoint,
            justification=f"PARSER ERROR: {str(e)}"
        )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_model(
    model_name: str,
    question_text: str,
    prompt_style_key: str,
    scale_range: List[int],
    temperature: float = 0.0
) -> Tuple[Optional[SurveyAnswer], str]:
    """
    Creates the final prompt and calls the appropriate client.
    """
    validate_scale(scale_range)
    
    config = MODEL_CONFIG[model_name]
    style_prompt = prompt_templates[prompt_style_key]
    scale_str = f"(Scale from {scale_range[0]} to {scale_range[1]})"

    final_prompt = f\"\"\"{style_prompt}\n
Question: {question_text}\n
{scale_str}\n
Please provide your response in JSON format:\n
{{"rating": <number>, "justification": "<explanation>"}}\"\"\"

    raw_text = ""
    try:
        # ------------------------------------------------------------------
        # 1) OpenAI => AsyncOpenAI usage
        # ------------------------------------------------------------------
        if model_name == "OpenAI":
            client = AsyncOpenAI(api_key=config["api_key"])
            response = await client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": final_prompt}],
                temperature=temperature,
                max_tokens=512,
            )
            if hasattr(response, 'usage'):
                cost_tracker[model_name] += (
                    response.usage.prompt_tokens + response.usage.completion_tokens
                )
            raw_text = response.choices[0].message.content
            parsed = safe_parse_survey_answer(raw_text, scale_range)
            return parsed, raw_text

        # ------------------------------------------------------------------
        # 2) Grok => same AsyncOpenAI approach but with base_url
        # ------------------------------------------------------------------
        elif model_name == "Grok":
            base_url = config["base_url"]
            client = AsyncOpenAI(api_key=config["api_key"], base_url=base_url)
            response = await client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": final_prompt}],
                temperature=temperature,
                max_tokens=512,
            )
            if hasattr(response, 'usage'):
                cost_tracker[model_name] += (
                    response.usage.prompt_tokens + response.usage.completion_tokens
                )
            raw_text = response.choices[0].message.content
            parsed = safe_parse_survey_answer(raw_text, scale_range)
            return parsed, raw_text
        # ------------------------------------------------------------------
        # 3) Claude => Using AsyncAnthropic client
        # ------------------------------------------------------------------
        elif model_name == "Claude":
            client = AsyncAnthropic(api_key=config["api_key"])
            
            response = await client.messages.create(
                model=config["model"],
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
            )
            
            raw_text = ""
            if hasattr(response, 'content'):
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        raw_text += content_block.text
            
            if hasattr(response, 'usage'):
                cost_tracker[model_name] += (
                    response.usage.input_tokens + response.usage.output_tokens
                )
            
            parsed = safe_parse_survey_answer(raw_text, scale_range)
            return parsed, raw_text
        # ------------------------------------------------------------------
        # 4) Llama or DeepSeek 
        # ------------------------------------------------------------------
      
        elif model_name in ["Llama", "DeepSeek"]:
            llama_client = LlamaAPI(config["api_key"])
            
            request_data = {
                "model": config["model"],
                "messages": [
                    {"role": "user", "content": final_prompt}
                ],
                "stream": False
            }
            
            def run_llama():
                try:
                    response = llama_client.run(request_data)
                    raw_text = response.json()["choices"][0]["message"]["content"]
                    
                    if "usage" in response.json():
                        cost_tracker[model_name] += response.json()["usage"].get("total_tokens", 0)
                    
                    return safe_parse_survey_answer(raw_text, scale_range), raw_text
                    
                except Exception as e:
                    logger.error(f"Llama API error: {str(e)}")
                    return None, str(e)
            
            loop = asyncio.get_running_loop()
            parsed, raw_text = await loop.run_in_executor(None, run_llama)
            return parsed, raw_text

        else:
            return None, f"Error: Unknown model {model_name}"

    except Exception as e:
        logger.error(f"Error calling {model_name}: {str(e)}")
        raise

async def call_model_api(model_name, question_text, prompt_style_key, scale_range, temperature=0.0) -> Dict:
    start_time = time.time()
    parsed, raw_response = await call_model(model_name, question_text, prompt_style_key, scale_range, temperature)
    
    if parsed is None:
        min_scale, max_scale = scale_range
        midpoint = (min_scale + max_scale) / 2
        parsed = SurveyAnswer(
            numeric_score=midpoint,
            justification=f"API ERROR: {raw_response[:100]}..."
        )
    
    elapsed = time.time() - start_time
    return {
        "model_name": model_name,
        "numeric_score": parsed.numeric_score,
        "label": parsed.label,
        "justification": parsed.justification,
        "raw_response": raw_response,
        "duration": elapsed
    }
