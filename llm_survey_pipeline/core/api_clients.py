"""API client implementations for LLM Survey Pipeline"""
import asyncio
import logging
from typing import Tuple, Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from llamaapi import LlamaAPI

from llm_survey_pipeline.config import MODEL_CONFIG, prompt_templates
from llm_survey_pipeline.models import SurveyAnswer, validate_scale
from llm_survey_pipeline.utils.cost_tracking import cost_tracker
from llm_survey_pipeline.core.parsers import safe_parse_survey_answer

logger = logging.getLogger(__name__)


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

    final_prompt = f"""{style_prompt}

Question: {question_text}
{scale_str}

Please provide your response in JSON format:
{{"rating": <number>, "justification": "<explanation>"}}"""

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
            
            # Get content from response
            raw_text = ""
            if hasattr(response, 'content'):
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        raw_text += content_block.text
            
            # Track tokens if available
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
                    
                    # Track tokens if available
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
        raise  # Let retry handle it


async def call_model_api(model_name, question_text, prompt_style_key, scale_range, temperature=0.0) -> Dict:
    """
    Wrapper to measure time & return a dict of info
    """
    import time
    start_time = time.time()
    parsed, raw_response = await call_model(model_name, question_text, prompt_style_key, scale_range, temperature)
    
    # If parsing failed, create a fallback SurveyAnswer with scale midpoint
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
        "numeric_score": parsed.numeric_score,  # Will always have a value 
        "label": parsed.label,
        "justification": parsed.justification,
        "raw_response": raw_response,
        "duration": elapsed
    }