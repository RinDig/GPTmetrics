# LLM Survey Aggregator

## Overview

This project uses a Jupyter Notebook (`survey_aggregator.ipynb`) to systematically administer psychological and political surveys to various Large Language Models (LLMs). It is designed to test the responses of different models under various prompting conditions (e.g., adopting a specific political persona) across several well-established psychological scales.

![Graph](https://github.com/user-attachments/assets/186a4c49-c197-4f7a-88a7-f0776d157538)


The primary goal is to gather data on how different LLMs respond to nuanced, politically-charged, or value-laden questions, providing a framework for analyzing potential biases, personality traits, or cognitive styles embedded in these models.

## Features

- **Multi-Model Support**: Easily configurable to run surveys on multiple LLMs, including:
  - OpenAI (e.g., `gpt-4o`)
  - Anthropic (e.g., `claude-3-5-sonnet`)
  - Llama API-compatible models (e.g., `llama3.1-70b`)
  - Grok (`grok-2-latest`)
  - DeepSeek (`deepseek-v3`)
- **Diverse Psychological Scales**: Includes a variety of pre-configured survey instruments:
  - **RWA (Right-Wing Authoritarianism)**: Measures authoritarian submission, aggression, and conventionalism.
  - **LWA (Left-Wing Authoritarianism)**: Measures authoritarianism from a left-wing perspective.
  - **MFQ (Moral Foundations Questionnaire)**: Assesses reliance on different moral foundations (Care, Fairness, Loyalty, Authority, Purity).
  - **NFC (Need for Cognition)**: Measures an individual's tendency to engage in and enjoy thinking.
- **Flexible Prompting**: Apply different "personas" or prompt styles to the models to test how their responses change based on the assigned identity (e.g., "minimal," "extreme liberal," "extreme conservative").
- **Asynchronous Processing**: Leverages `asyncio` to run API calls concurrently, significantly speeding up data collection.
- **Smart Rate Limiting**: Uses semaphores and queues to manage API calls efficiently, respecting rate limits for different providers (OpenAI, Anthropic, etc.).
- **Robust Parsing & Error Handling**: Includes a flexible parser to extract numeric scores and justifications from model responses. It has built-in retries (`tenacity`) and fallback mechanisms for failed API calls or parsing errors.
- **Detailed Output**: Generates clean, analyzable CSV files with raw responses, scored values (including reverse-scoring), and aggregated results for specific scales like the MFQ.

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    Install the required packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create Environment File**:
    Create a `.env` file in the root directory by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Open the `.env` file and add your API keys for the services you intend to use:
    ```
    OPENAI_API_KEY="your-openai-key"
    ANTHROPIC_API_KEY="your-anthropic-key"
    LLAMA_API_KEY="your-llamaapi-key"
    XAI_API_KEY="your-grok-key"
    DEEPSEEK_API_KEY="your-deepseek-key"
    ```

## How to Use

1.  **Configure the Experiment**:
    Open the `survey_aggregator.ipynb` notebook. Navigate to the **Global Parameters** cell.
    Modify the following lists to define the scope of your run:
    - `MODELS_TO_RUN`: A list of model names to test (e.g., `["OpenAI", "Claude"]`).
    - `SCALES_TO_RUN`: A list of psychological scales to administer (e.g., `["RWA", "LWA"]`).
    - `PROMPT_STYLES_TO_RUN`: A list of personas the models should adopt (e.g., `["minimal", "extreme_liberal"]`).
    - `NUM_CALLS_TEST`: The number of times to repeat each unique question-model-prompt combination.

2.  **Run the Notebook**:
    Execute the cells in the Jupyter Notebook from top to bottom. A progress bar will appear to show the status of the API calls.

## Output Files

After a successful run, the following files will be generated:

-   `unified_responses.csv`: The primary output file. It contains one row for every single API call made, including:
    -   Model name, prompt style, scale name, question ID, and question text.
    -   The raw text response from the model.
    -   The parsed numeric score and justification.
    -   The final `scored_value` after applying reverse-scoring logic where needed.
    -   Metadata like call duration.

-   `refusal_responses.csv`: A filtered version of the main results, containing only the rows where the model failed to provide a valid numeric answer. This is useful for analyzing model refusals or parsing issues.

-   `mfq_foundation_scores.csv`: If the `MFQ` scale is run, this file contains aggregated scores for each of the six moral foundations (Care, Equality, Proportionality, Loyalty, Authority, Purity), broken down by model and prompt style.

## Code Structure

The `survey_aggregator.ipynb` notebook is organized into the following main sections:

1.  **Imports and Setup**: Initializes libraries and loads environment variables.
2.  **Model and Prompt Configuration**: Defines the `MODEL_CONFIG`, `prompt_templates`, and the questions for each scale (`rwa_questions`, `lwa_questions`, etc.).
3.  **Core API and Parsing Logic**: Contains the `safe_parse_survey_answer`, `call_model`, and `call_model_api` functions that form the backbone of the data collection process.
4.  **Task Execution Engine**: The `process_tasks_in_chunks` function manages the asynchronous execution of all survey tasks.
5.  **Main Execution Block**: Sets the global parameters for the run, builds the task list, and initiates the execution.
6.  **Data Processing and Saving**: Once the results are collected, this section applies reverse-scoring, builds a Pandas DataFrame, and saves the `unified_responses.csv`.
7.  **Analysis Functions**: Includes helper functions to generate supplementary reports, like `save_refusal_responses` and `calculate_mfq_scores`.
