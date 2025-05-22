import asyncio
import pandas as pd
import logging
import nest_asyncio

# Apply nest_asyncio early, as in the notebook
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration imports
from config import (
    all_questions,
    SCALES_TO_RUN,
    MODELS_TO_RUN,
    PROMPT_STYLES_TO_RUN,
    NUM_CALLS_TEST
)

# LLM services imports
from llm_services import cost_tracker # cost_tracker is initialized here

# Task processing imports
from task_processing import process_tasks_in_chunks

# Data analysis imports
from data_analysis import (
    apply_reverse_score,
    save_refusal_responses,
    calculate_mfq_scores,
    calculate_rwa_scores,
    save_rwa_results,
    calculate_lwa_scores,
    save_lwa_results,
    calculate_average_scores
)

# Visualization imports
from visualization import (
    plot_authoritarian_scores,
    plot_avg_authoritarian_scores
)

async def main_async():
    logger.info("Starting the survey aggregation pipeline.")

    # 3. Build the tasks list
    tasks = []
    logger.info(f"Building tasks for scales: {SCALES_TO_RUN}, models: {MODELS_TO_RUN}, prompt_styles: {PROMPT_STYLES_TO_RUN}, num_calls: {NUM_CALLS_TEST}")
    for q in all_questions:
        if q["scale_name"] not in SCALES_TO_RUN:
            continue
        scale_name = q["scale_name"]
        question_id = q["id"]
        question_text = q["text"]
        scale_range = q["scale_range"]
        reverse_score = q.get("reverse_score", False)

        for model_name in MODELS_TO_RUN:
            for prompt_style in PROMPT_STYLES_TO_RUN:
                for run in range(1, NUM_CALLS_TEST + 1):
                    tasks.append({
                        "model_name": model_name,
                        "scale_name": scale_name,
                        "question_id": question_id,
                        "question_text": question_text,
                        "prompt_style": prompt_style,
                        "run_number": run,
                        "scale_range": scale_range,
                        "reverse_score": reverse_score,
                    })
    logger.info(f"Total tasks built: {len(tasks)}")
    
    # (Optionally limit tasks for quick testing - as in notebook)
    # tasks = tasks[:20] 
    # logger.info(f"Limiting tasks to {len(tasks)} for testing.")


    # 4. Execute tasks
    logger.info("Processing tasks...")
    results = await process_tasks_in_chunks(tasks)
    logger.info("Task processing completed.")

    # 5. Process and Save Initial Results
    df_results = pd.DataFrame(results)
    logger.info(f"Total results received: {len(df_results)}")

    if not df_results.empty:
        df_results["scored_value"] = df_results.apply(apply_reverse_score, axis=1)
        df_results.to_csv("unified_responses.csv", index=False)
        logger.info("Saved responses to unified_responses.csv")

        # Print summary info
        logger.info("\nProcessing Summary:")
        logger.info(f"Total responses: {len(df_results)}")
        logger.info(f"Token usage by model: {dict(cost_tracker)}") # cost_tracker from llm_services
        
        # Ensure 'numeric_score' column exists before trying to calculate response rate
        if 'numeric_score' in df_results.columns:
            resp_rate = df_results['numeric_score'].notna().sum() / len(df_results) * 100
            logger.info(f"Response rate: {resp_rate:.1f}%")
        else:
            logger.warning("'numeric_score' column not found in results. Cannot calculate response rate.")

        # 6. Save Refusal Responses
        save_refusal_responses(df_results, output_file="refusal_responses.csv")

        # 7. Perform Analyses
        if "MFQ" in SCALES_TO_RUN:
            logger.info("Calculating MFQ scores...")
            calculate_mfq_scores("unified_responses.csv") # This function saves its own CSV and prints
        
        if "RWA" in SCALES_TO_RUN:
            logger.info("Calculating RWA scores...")
            rwa_results = calculate_rwa_scores(df_results)
            save_rwa_results(rwa_results, "rwa_results.csv")
            logger.info("\nRWA Scores by Model and Prompt:")
            logger.info(rwa_results)
            
            logger.info("Calculating average RWA scores...")
            rwa_avg_scores = calculate_average_scores(df_results, "RWA", 34)
            rwa_avg_scores.to_csv("rwa_avg_results.csv", index=False)
            logger.info("Saved RWA average scores to rwa_avg_results.csv")

        if "LWA" in SCALES_TO_RUN:
            logger.info("Calculating LWA scores...")
            lwa_results = calculate_lwa_scores(df_results)
            save_lwa_results(lwa_results, "lwa_results.csv")
            logger.info("\nLWA Scores by Model and Prompt:")
            logger.info(lwa_results)

            logger.info("Calculating average LWA scores...")
            lwa_avg_scores = calculate_average_scores(df_results, "LWA", 39)
            lwa_avg_scores.to_csv("lwa_avg_results.csv", index=False)
            logger.info("Saved LWA average scores to lwa_avg_results.csv")

        # 8. Generate Visualizations
        if "RWA" in SCALES_TO_RUN and "LWA" in SCALES_TO_RUN:
            logger.info("Generating authoritarian scores plot...")
            plot_authoritarian_scores()
            logger.info("Generating average authoritarian scores plot...")
            plot_avg_authoritarian_scores()
        else:
            logger.info("Skipping authoritarian plots as both RWA and LWA are not in SCALES_TO_RUN.")

    else:
        logger.warning("No results were processed. Skipping data analysis and visualization.")

    logger.info("Survey aggregation pipeline finished.")

if __name__ == "__main__":
    asyncio.run(main_async())
