# The Ethics Engine: Measuring Moral and Ideological Bias in LLMs

This repository hosts a **modular Python pipeline** designed to detect, measure, and visualize **moral** and **ideological** biases in Large Language Models (LLMs). It allows users to run questionnaire-based tests against various LLMs and analyze their responses based on established psychometric frameworks.

---

## Project Overview

- **Goal**: To provide a systematic way to collect and interpret LLM outputs in response to psychological surveys, using scales like Moral Foundations Theory (MFQ), Right-Wing Authoritarianism (RWA), and Left-Wing Authoritarianism (LWA).
- **Methodology**: The pipeline sends survey questions to configured LLMs using different prompt styles. The responses are then collected, scored, and analyzed.
- **Outcome**: The project generates CSV files with raw and scored data, as well as visualizations to help understand the moral and ideological leanings of different LLMs under various prompting conditions.

---

## Modular Structure

The project has been refactored from a Jupyter Notebook into a set of Python modules for better organization and reusability:

-   `main.py`: The main script to run the entire pipeline.
-   `config.py`: Contains all configurations, including API key environment variable names, model settings, prompt templates, survey questions, and global run parameters (e.g., which scales, models, and prompt styles to use).
-   `llm_services.py`: Handles interactions with various LLM APIs (OpenAI, Anthropic, LlamaAPI for Llama and DeepSeek, Grok via XAI). Includes response parsing and cost tracking.
-   `task_processing.py`: Manages the asynchronous execution of survey tasks, including chunking and rate limiting for different API providers.
-   `data_analysis.py`: Performs scoring for different scales (MFQ, RWA, LWA), handles reverse scoring, and identifies refusal responses.
-   `visualization.py`: Generates plots to visualize the analysis results, such as authoritarian scores.

---

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/RinDig/GPTmetrics.git
    cd GPTmetrics
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    Ensure you have Python 3.8+ installed. Then, install the required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Keys**:
    Create a `.env` file in the root directory of the project. This file will store your API keys. Add your keys to this file, following the template below:

    ```env
    OPENAI_API_KEY=your_openai_key_here
    ANTHROPIC_API_KEY=your_anthropic_key_here
    LLAMA_API_KEY=your_llamaapi_key_here
    XAI_API_KEY=your_xai_api_key_for_grok_here
    DEEPSEEK_API_KEY=your_deepseek_api_key_here
    ```
    The pipeline loads these keys using `python-dotenv`.

---

## Running the Pipeline

1.  **Configure the Run (Optional)**:
    Before running, you might want to adjust parameters in `config.py`. Key variables you can modify include:
    -   `SCALES_TO_RUN`: A list of scales to include in the run (e.g., `["LWA", "RWA", "MFQ"]`).
    -   `MODELS_TO_RUN`: A list of LLMs to test (e.g., `["OpenAI", "Claude", "Grok"]`). Ensure corresponding API keys are in your `.env` file.
    -   `PROMPT_STYLES_TO_RUN`: A list of prompt styles to use for each question (e.g., `["minimal", "extreme_liberal"]`).
    -   `NUM_CALLS_TEST`: The number of times each question/prompt/model combination will be run.

2.  **Execute the Main Script**:
    Run the pipeline using:
    ```bash
    python main.py
    ```
    The script will process the tasks, perform analyses, and generate output files.

---

## Output Files

The pipeline generates several output files in the root directory:

-   `unified_responses.csv`: Contains all raw and scored responses from the LLMs, including metadata about the model, prompt style, question, and run number.
-   `refusal_responses.csv`: A subset of `unified_responses.csv` that logs instances where models refused to answer or provided responses that could not be parsed into a valid score.
-   `mfq_foundation_scores.csv`: (If MFQ is run) Contains aggregated scores for each moral foundation, per model and prompt style.
-   `rwa_results.csv`: (If RWA is run) Contains aggregated total and normalized RWA scores.
-   `lwa_results.csv`: (If LWA is run) Contains aggregated total and normalized LWA scores.
-   `rwa_avg_results.csv`: (If RWA is run) Contains average RWA scores per question.
-   `lwa_avg_results.csv`: (If LWA is run) Contains average LWA scores per question.
-   `authoritarian_scores_plot.png`: (If RWA & LWA are run) A plot visualizing normalized RWA and LWA scores.
-   `avg_authoritarian_scores_plot.png`: (If RWA & LWA are run) A plot visualizing average RWA and LWA scores per question.

Logs are printed to the console during execution.

---

## Future Directions

-   Add more psychometric scales.
-   Support local models for offline testing.
-   Expand output to user-friendly dashboards.

---

## Citation

If you use this pipeline or its ideas, please consider citing the related work:
```
@misc{vanclief2025ethicsengine,
  title     = {The Ethics Engine: Building a Modular Pipeline to Uncover AI Bias and Moral Alignments},
  author    = {J.E. Van Clief},
  year      = {2025},
  url       = {https://github.com/RinDig/GPTmetrics}
}
```
