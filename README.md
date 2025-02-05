# LLM Political Survey Analysis

A computational social science research project investigating how different AI language models respond to standardized political and moral survey questions across various ideological perspectives.

![Series Graph MFQ](https://github.com/user-attachments/assets/e5f14f85-e6dd-436f-9b60-111dc50fd33f)

## Project Overview

This research systematically analyzes how Large Language Models (LLMs) interpret and respond to political and moral questions when prompted with different ideological perspectives. By using standardized psychometric scales and deliberate prompt engineering, we explore AI moral foundations and potential political biases.

### Research Questions
- How do different LLMs interpret moral and political survey questions?
- Can LLMs effectively role-play different political perspectives?
- Do responses vary systematically across different prompting styles?
- What biases or patterns emerge in moral foundation scores?

### Implemented Surveys
- **Moral Foundations Questionnaire (MFQ)**
  - Measures moral intuitions across care, fairness, loyalty, authority, and purity
- **Right-Wing Authoritarianism Scale (RWA)**
  - Assesses authoritarian submission, aggression, and conventionalism
- **Left-Wing Authoritarianism Scale (LWA)**
  - Measures anti-hierarchical aggression, top-down censorship, and anti-conventionalism

### Supported Models
- OpenAI (GPT4o and later) 
- Anthropic (Claude)
- Llama (DeepSeek included as well as all other Llama framework models)
- Grok

## Technical Architecture

### Tech Stack
- **Implementation**: Python 3.8+ in Jupyter Notebook
- **Data Storage**: CSV-based
- **Key Libraries**:
  ```
  pandas         # Data processing
  asyncio        # Concurrent API calls
  openai         # OpenAI API client
  anthropic      # Claude API client
  llamaapi       # Llama API client
  pydantic       # Data validation
  plotly/seaborn # Visualization
  ```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/RinDig/PsychoMetricGPT.git
cd llm-political-survey
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
export OPENAI_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'
export LLAMA_API_KEY='your-key-here'
export XAI_API_KEY='your-key-here'
export DEEPSEEK_API_KEY='your-key-here'
```

## Usage

The project is implemented in a Jupyter notebook (`survey_aggregator.ipynb`) with these main components:

### 1. Configuration
```python
SCALES_TO_RUN = ["MFQ"]  # Options: ["MFQ", "RWA", "LWA"]
PROMPT_STYLES = ["neutral", "extreme_liberal", "extreme_republican"]
MODELS_TO_RUN = ["OpenAI", "Claude", "Llama", "Grok", "DeepSeek"]
```

### 2. Data Collection
- Async API calls to multiple LLMs
- Rate limit handling and error management
- Structured response parsing
- Response validation and scoring

### 3. Analysis Pipeline
- Foundation score calculation
- Political alignment analysis
- Statistical comparisons
- Visualization generation

## Output Files

The analysis generates several key outputs:
- `unified_responses.csv`: Raw LLM responses
- `mfq_foundation_scores.csv`: Processed moral foundation scores
- Visualization plots comparing model responses

## Contributing

Contributions are welcome! Areas for potential improvement:
- Additional psychometric scales
- New LLM integrations
- Enhanced visualization options
- Statistical analysis methods


## Citation

If you use this code in your research, please cite:

```bibtex
  author = {J.E. Van Clief},
  title = {LLM Political Survey Analysis},
  year = {2024},
  url = {[https://github.com/yourusername/llm-political-survey](https://github.com/RinDig/PsychoMetricGPT)}
}
```

## Acknowledgments

- Moral Foundations Questionnaire (MFQ) by Graham, Haidt, & Nosek
- Right-Wing Authoritarianism Scale by Altemeyer
- Left-Wing Authoritarianism Scale by Conway et al.
