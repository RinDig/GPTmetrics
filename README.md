# The Ethics Engine: Measuring Moral and Ideological Bias in LLMs

This repository hosts a **modular pipeline** to detect, measure, and visualize **moral** and **ideological** biases in Large Language Models (LLMs).

You can use it to run **direct (questionnaire-based)** tests and **indirect (embedding-based)** checks, helping you see how models respond to various scales and prompts.

---
![WhatsApp Image 2025-02-15 at 19 37 09_6dc12657](https://github.com/user-attachments/assets/ecc5a3cc-b320-4420-bf66-1079bc737e57)

## Overview

- **Goal**: Give researchers, policymakers, and other users a way to collect and interpret LLM outputs against **psychometric frameworks** like Moral Foundations Theory, Right-Wing Authoritarianism, and more.
- **Key Idea**: Gather data from LLMs under different prompt styles, then run both **survey scoring** and **embedding analysis** to find underlying patterns.
- **Outcome**: Produce clear findings about moral and ideological leanings, which can inform AI audits, model improvements, and policy steps.

---

## Features

- **Direct Questionnaire**: Prompts LLMs with standard survey items and uses well-known scoring methods.
- **Indirect Embedding**: Analyzes text embeddings to spot hidden patterns or moral signals in responses.
- **Modular Setup**: Add new scales, tweak prompt styles, or plug in new models with ease.
- **Clear Outputs**: Generates CSV files and charts for quick review.

---

## Getting Started

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Set API Keys**  
   ```bash
   export OPENAI_API_KEY=your_openai_key
   export ANTHROPIC_API_KEY=your_anthropic_key
   ```
3. **Configure the Pipeline**  
   - Edit `config.yaml` or `config.json` to choose scales (MFQ, RWA, LWA) and define prompt styles.
4. **Run the Notebook**  
   - Open `survey_aggregator.ipynb` in Jupyter.
   - Adjust settings if needed.
   - Run all cells.

---

## Pipeline Flow

1. **Config Load**  
   - Reads user-chosen scales, model keys, and prompt strategies.
2. **Data Collection**  
   - Sends standardized prompts to chosen LLMs (GPT, Claude, Llama, etc.).
   - Handles retries and stores raw outputs.
3. **Analysis**  
   - **Direct**: Scores responses against known scales (e.g., MFQ sub-scales).
   - **Indirect**: Converts responses into embeddings, then runs clustering or PCA to find bias signals.
4. **Reporting**  
   - Creates CSV files with raw scores and summary stats.
   - Generates plots to compare results across models and prompts.

---

## Example Usage

```python
# In survey_aggregator.ipynb
SCALES_TO_RUN = ["MFQ", "RWA", "LWA"]
PROMPT_STYLES = ["neutral", "left_biased", "right_biased"]
MODELS_TO_RUN = ["OpenAI", "Claude", "Llama"]

# Run data collection
collect_data()

# Run analysis and show charts
analyze_responses()
plot_results()
```

---

## Results

- **Quantitative Scores**: Table of mean scores for each moral foundation or authoritarian scale per model.
- **Visualization**: Radar plots or bar charts showing how each model leans under different prompts.
- **Embedding Insights**: PCA or clustering plots that show hidden groupings in text outputs.

---

## Future Directions

- Add more scales (e.g., **Dark Triad**, **Schwartz Values**).
- Support local models for offline testing.
- Expand output to user-friendly dashboards for policymakers.
- Refine prompt strategies to test even more nuanced biases.

---

## Citation

If you use this pipeline or its ideas, please cite the related dissertation:

```
@misc{vanclief2025ethicsengine,
  title     = {The Ethics Engine: Building a Modular Pipeline to Uncover AI Bias and Moral Alignments},
  author    = {J.E. Van Clief},
  year      = {2025},
  url       = {https://github.com/RinDig/GPTmetrics}
}
