import json
import re
import logging
from typing import List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
    label: str = ""
    justification: str = ""

def safe_parse_survey_answer(response_text: str, scale_range: List[int]) -> Optional[SurveyAnswer]:
    """
    Simplified parser that handles both structured and unstructured responses.
    Prioritizes finding valid numeric scores within the scale range.
    """
    response_text = str(response_text).strip().lower()
    min_scale, max_scale = scale_range
    
    def is_valid_score(num: float) -> bool:
        return min_scale <= num <= max_scale

    try:
        simple_match = re.search(r'(?:rating|score):\s*(-?\d+(?:\.\d+)?)', response_text)
        if simple_match:
            num = float(simple_match.group(1))
            if is_valid_score(num):
                explanation = response_text[simple_match.end():].strip()
                return SurveyAnswer(
                    numeric_score=num,
                    justification=explanation or response_text
                )

        try:
            data = json.loads(response_text)
            if 'rating' in data and is_valid_score(float(data['rating'])):
                return SurveyAnswer(
                    numeric_score=float(data['rating']),
                    justification=data.get('justification', '')
                )
        except json.JSONDecodeError:
            pass

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