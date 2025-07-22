# LLM Survey Aggregator

## Overview

This project uses a python pipline to systematically administer psychological and political surveys to various Large Language Models (LLMs). It is designed to test the responses of different models under various prompting conditions (e.g., adopting a specific political persona) across several well-established psychological scales.

![WhatsApp Image 2025-06-10 at 17 42 25_797f3260](https://github.com/user-attachments/assets/c79330df-9e96-4f25-a396-8135dcce1f2f)



The primary goal is to gather data on how different LLMs respond to nuanced, politically-charged, or value-laden questions, providing a framework for analyzing potential biases, personality traits, or cognitive styles embedded in these models.

## Features

- **Modular Architecture**: Clean separation of concerns with organized modules
- **Web Dashboard**: Interactive Streamlit dashboard for easy configuration and visualization
- **CLI Interface**: Command-line interface for automated runs
- **Scalable Design**: Easy to add new models, scales, or prompt templates
- **Real-time Progress**: Visual progress tracking during survey execution
- **Advanced Analytics**: Built-in analysis and visualization tools

## Directory Structure

```
llm_survey_pipeline/
├── config/              # Configuration modules
│   ├── models.py        # Model configurations
│   ├── prompts.py       # Prompt templates (preserved exactly)
│   └── scales.py        # Scale templates (RWA, LWA, MFQ, NFC - preserved exactly)
├── core/                # Core functionality
│   ├── api_clients.py   # API client implementations
│   ├── parsers.py       # Response parsing logic
│   ├── processors.py    # Task processing engine
│   └── validators.py    # Data validation
├── models/              # Data models
│   └── data_models.py   # Pydantic models
├── utils/               # Utilities
│   ├── cost_tracking.py # Token/cost tracking
│   └── analysis.py      # Analysis functions
├── dashboard/           # Web dashboard
│   └── app.py          # Streamlit application
├── data/outputs/        # Output directory
├── main.py             # CLI interface
└── requirements.txt
```

## Installation

1. Clone or copy the `llm_survey_pipeline` directory
2. Create a virtual environment:
   ```bash
   cd llm_survey_pipeline
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY="your-openai-key"
   ANTHROPIC_API_KEY="your-anthropic-key"
   LLAMA_API_KEY="your-llamaapi-key"
   XAI_API_KEY="your-grok-key"
   DEEPSEEK_API_KEY="your-deepseek-key"
   ```

## Usage

### Web Dashboard

Run the dashboard for an interactive experience:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- Interactive model, scale, and prompt selection
- Real-time progress tracking
- Visual analytics and charts
- Data export functionality
- Configuration viewing

### Command Line Interface

For automated runs or integration with scripts:

```bash
python main.py --scales RWA LWA --models OpenAI Claude --prompts minimal extreme_liberal --runs 2
```

Options:
- `--scales`: List of scales to run (RWA, RWA2, LWA, MFQ, NFC)
- `--models`: List of models to test (OpenAI, Claude, Grok, Llama, DeepSeek)
- `--prompts`: Prompt styles to use
- `--runs`: Number of runs per question
- `--temperature`: Model temperature (default: 0.0)
- `--output-dir`: Output directory for results

## Adding New Components

### Adding a New Scale

1. Edit `config/scales.py`
2. Add your scale questions following the existing format:
   ```python
   new_scale_questions = [
       {"scale_name": "NEW", "id": "NEW_1", "text": "Question text", 
        "scale_range": [1,7], "reverse_score": False},
       # ... more questions
   ]
   ```
3. Add to `all_scales` list

### Adding a New Model

1. Edit `config/models.py`
2. Add model configuration:
   ```python
   "NewModel": {
       "client": "openai",  # or "anthropic", "llamaapi"
       "model": "model-name",
       "api_key": os.getenv("NEW_MODEL_API_KEY"),
   }
   ```

### Adding a New Prompt Style

1. Edit `config/prompts.py`
2. Add prompt template:
   ```python
   "new_style": "Your prompt template here..."
   ```

## Output Files

- `unified_responses.csv`: All survey responses with scores
- `refusal_responses.csv`: Responses where models refused/failed
- `mfq_foundation_scores.csv`: MFQ foundation analysis (if MFQ scale is run)

## API Rate Limits

The system includes intelligent rate limiting:
- OpenAI/Grok: 3 concurrent calls, 1 second between chunks
- Anthropic: 5 concurrent calls, 0.5 seconds between chunks
- Llama/DeepSeek: 10 concurrent calls, 0.2 seconds between chunks

## Notes

- All scale templates and prompt templates are preserved exactly as in the original notebook
- The modular structure makes it easy to extend and maintain
- The dashboard provides a user-friendly interface while maintaining all original functionality
- Progress bars and real-time updates keep you informed during long runs
