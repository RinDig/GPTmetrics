# LLM Survey Aggregator

## Overview

This project uses a python pipline to systematically administer psychological and political surveys to various Large Language Models (LLMs). It is designed to test the responses of different models under various prompting conditions (e.g., adopting a specific political persona) across several well-established psychological scales.

<img width="1004" height="718" alt="image" src="https://github.com/user-attachments/assets/013f2760-5f73-4846-a720-8c181ce18d49" />


The primary goal is to gather data on how different LLMs respond to nuanced, politically-charged, or value-laden questions, providing a framework for analyzing potential biases, personality traits, or cognitive styles embedded in these models.
# LLM Psychological Assessment Survey

A comprehensive framework for evaluating psychological and political dimensions of Large Language Models (LLMs) using established psychometric scales.

## Overview

This project surveys multiple state-of-the-art LLMs (OpenAI GPT models, Claude, Grok, Llama, DeepSeek) using validated psychological assessment instruments to measure their responses across different political biases and psychological dimensions. The framework provides insights into how these models exhibit patterns related to authoritarianism, moral foundations, and cognitive preferences.

## Features

- **Multi-Model Support**: Survey OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, xAI Grok-2, Meta Llama 3.1-70B, and DeepSeek-v3 models
- **Comprehensive Assessment Scales**:
  - Right-Wing Authoritarianism (RWA): 34 questions
  - Right-Wing Authoritarianism v2 (RWA2): 22 questions  
  - Left-Wing Authoritarianism (LWA): 39 questions
  - Moral Foundations Questionnaire (MFQ): 36 questions
  - Need for Cognition (NFC): 45 questions
- **Dual Response Formats**: 
  - Numeric prompts: Direct numerical scale responses
  - Text prompts: Verbal Likert-scale responses with automatic conversion
- **Flexible Prompt Styles**: Test models with different political bias framings (extreme liberal, moderate liberal, neutral, moderate conservative, extreme conservative)
- **Robust Processing Pipeline**: 
  - Async API calls with retry logic and exponential backoff
  - Separate rate limiting per API provider
  - Advanced response parsing with multiple fallback mechanisms
  - Text-to-number conversion for verbal responses
- **Automated Analysis**: Response aggregation, reverse scoring, MFQ foundation score calculation, and refusal tracking

## Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- API keys for the LLM providers you want to survey

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt-metrics.git
cd gpt-metrics
```

2. Install dependencies:
```bash
pip install pandas seaborn matplotlib python-dotenv nest_asyncio openai anthropic llamaapi tenacity tqdm notebook
```

3. Set up API keys:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LLAMA_API_KEY=your_llama_key
XAI_API_KEY=your_xai_key
DEEPSEEK_API_KEY=your_deepseek_key
```

### Usage

1. Launch Jupyter Notebook:
```bash
jupyter notebook survey_aggregator.ipynb
```

2. Configure survey parameters in the notebook:
   - `SCALES_TO_RUN`: Select psychological scales to administer
   - `MODELS_TO_RUN`: Choose which LLMs to survey
   - `PROMPT_STYLES_TO_RUN`: Define political bias prompts
   - `NUM_CALLS_TEST`: Set number of repetitions per question

3. Run all cells to execute the survey and generate results

## Output Files

The framework generates three CSV files:

- **`unified_responses.csv`**: Complete survey responses from all models
- **`mfq_foundation_scores.csv`**: Aggregated Moral Foundations scores
- **`refusal_responses.csv`**: Instances where models refused or failed to answer

## Configuration

Key parameters can be adjusted in the notebook:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TEMPERATURE` | Model temperature setting | 1 |
| `MAX_CONCURRENT_CALLS` | API call concurrency limit | 5 |
| `NUM_CALLS_TEST` | Repetitions per question | 20 |
| `SCALES_TO_RUN` | Which psychological scales to survey | ["LWA", "RWA2"] |
| `PROMPT_STYLES_TO_RUN` | Political bias prompts to use | ["neutral", "extreme_liberal", "extreme_conservative"] |
| `MODELS_TO_RUN` | LLM models to query | ["OpenAI", "Claude", "Grok"] |

## Project Structure

```
gpt-metrics/
├── survey_aggregator.ipynb   # Main notebook with survey logic
├── .env                       # API keys (create this)
├── README.md                  # This file
└── output/                    # Generated CSV files (created on run)
    ├── unified_responses.csv
    ├── mfq_foundation_scores.csv
    └── refusal_responses.csv
```

## Psychological Scales

### Right-Wing Authoritarianism (RWA)
- **RWA**: 34-item scale measuring submission to authority, aggression toward outgroups, and conventionalism
- **RWA2**: 22-item revised version with updated wording

### Left-Wing Authoritarianism (LWA)
39-item scale measuring left-oriented authoritarian attitudes, including anti-hierarchical aggression and top-down censorship

### Moral Foundations Questionnaire (MFQ)
36-item scale assessing six moral foundations:
- Care/Harm
- Equality
- Proportionality  
- Loyalty/Betrayal
- Authority/Subversion
- Purity/Sanctity

### Need for Cognition (NFC)
45-item scale measuring tendency to engage in and enjoy effortful cognitive activities

## API Rate Limits

The framework includes built-in rate limiting and retry logic:
- Automatic retry with exponential backoff
- Concurrent request management
- Graceful handling of API errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llm_psychological_assessment,
  title = {LLM Psychological Assessment Survey},
  year = {2024},
  url = {https://github.com/yourusername/gpt-metrics}
}
```

## Acknowledgments

This project uses established psychological scales from peer-reviewed research. Please refer to the original publications for scale validation and psychometric properties.

## Contact

For questions or support, please open an issue on GitHub.onality
- Progress bars and real-time updates keep you informed during long runs
