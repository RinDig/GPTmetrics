import os

MODEL_CONFIG = {
    "OpenAI": {
        "client": "openai",
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    "Claude": {
        "client": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
    "Llama": {
        "client": "llamaapi",
        "model": "llama3.1-70b",
        "api_key": os.getenv("LLAMA_API_KEY"),
    },
    "Grok": {
        "client": "openai",
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