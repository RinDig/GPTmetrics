"""Model configuration for LLM Survey Pipeline"""
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_CONFIG = {
    "OpenAI": {
        "client": "openai",
        "model": "gpt-4o",  # or "gpt-4o-2024-08-06" if that is valid
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "Claude": {
        "client": "anthropic",
        "model": "claude-3-5-sonnet-20241022",  # or "claude-instant-1", etc.
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
    "Llama": {
        "client": "llamaapi",
        "model": "llama3.1-70b",
        "api_key": os.getenv("LLAMA_API_KEY"),
    },
    "Grok": {
        "client": "openai",  # using openai with custom base_url
        "model": "grok-2-latest",
        "api_key": os.getenv("XAI_API_KEY"),
        "base_url": "https://api.x.ai/v1",
    },
    "DeepSeek": {
        "client": "llamaapi",
        "model": "deepseek-v3",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
    },
}