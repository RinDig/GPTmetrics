"""Response parsing logic for LLM Survey Pipeline"""
import re
import json
import logging
from typing import Optional, List
from llm_survey_pipeline.models import SurveyAnswer

logger = logging.getLogger(__name__)


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