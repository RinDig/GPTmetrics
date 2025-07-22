"""Data models for LLM Survey Pipeline"""
from typing import List
from pydantic import BaseModel


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
    justification: str = ""