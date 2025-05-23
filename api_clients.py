import asyncio
import logging
import time
from typing import Tuple, Optional, List, Dict

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from llamaapi import LlamaAPI
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import defaultdict

from configs.model_config import MODEL_CONFIG
from configs.prompt_config import PROMPT_TEMPLATES
from data_processing.parsing import safe_parse_survey_answer, SurveyAnswer, validate_scale

logger = logging.getLogger(__name__)
cost_tracker = defaultdict(int)

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
    style_prompt = PROMPT_TEMPLATES[prompt_style_key]
    scale_str = f"(Scale from {scale_range[0]} to {scale_range[1]})"

    final_prompt = f"""{style_prompt}

Question: {question_text}
{scale_str}

Please provide your response in JSON format:
{{"rating": <number>, "justification": "<explanation>"}}"""

    raw_text = ""
    try:
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
                    api_response = llama_client.run(request_data)
                    resp_json = api_response.json()
                    raw_text_content = resp_json["choices"][0]["message"]["content"]
                    if "usage" in resp_json:
                        cost_tracker[model_name] += resp_json["usage"].get("total_tokens", 0)
                    return safe_parse_survey_answer(raw_text_content, scale_range), raw_text_content
                except Exception as e_llama:
                    logger.error(f"Llama API error: {str(e_llama)}")
                    return None, str(e_llama)
            
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
        "duration": elapsed,
        "cost_tracker": dict(cost_tracker) # Include current cost tracker state
    } 