import asyncio
import pandas as pd
import logging
import os
from dotenv import load_dotenv
import nest_asyncio
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, spearmanr

# Configuration Imports
from configs.app_config import (
    SCALES_TO_RUN, PROMPT_STYLES_TO_RUN, NUM_CALLS_TEST, MODELS_TO_RUN,
    TEMPERATURE, MAX_CONCURRENT_CALLS,
    UNIFIED_RESPONSES_CSV_PATH, SENTENCE_TRANSFORMER_MODEL_NAME,
    CCR_TEMP_DIR, SPEECH_MODEL_MAPPING, HOLISTIC_SPEECHES, OVERALL_GT_DATA
)
from llm_interface.api_clients import cost_tracker # Import cost_tracker

# Data Processing Imports
from data_processing.task_builder import build_tasks
from data_processing.parsing import SurveyAnswer # Assuming SurveyAnswer might be useful elsewhere or for type hinting

# LLM Interface Imports
# call_model_api is used by task_processor

# Utility Imports
from utils.task_processor import process_tasks_in_chunks

# Analysis Imports
from analysis.scoring import apply_reverse_score
from analysis.refusal_analysis import save_refusal_responses
from analysis.mfq_analysis import calculate_mfq_scores
from analysis.rwa_analysis import calculate_rwa_scores
from analysis.lwa_analysis import calculate_lwa_scores
from analysis.average_score_analysis import calculate_average_scores
from analysis.ccr.utils import (
    load_and_filter_data_for_ccr, 
    get_scale_definition_for_ccr,
    ensure_ccr_temp_dir
)
from analysis.ccr.analyzer import (
    run_item_analysis_ccr, 
    run_corpus_or_speech_analysis_ccr
)

# Plotting Imports
from plotting.authoritarian_plots import (
    plot_authoritarian_scores,
    plot_avg_authoritarian_scores,
    plot_separate_total_authoritarian_scores
)

# Apply nest_asyncio early if needed for Jupyter-like environments or specific async patterns
nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_survey_and_initial_processing():
    logger.info("Building tasks...")
    tasks = build_tasks()
    # Optionally limit tasks for quick testing: tasks = tasks[:20]
    logger.info(f"Built {len(tasks)} tasks.")

    if not tasks:
        logger.warning("No tasks were built. Check configurations in app_config.py and scale_definitions.py.")
        return None

    logger.info("Processing tasks...")
    results = await process_tasks_in_chunks(tasks)
    logger.info(f"Finished processing tasks. Total results: {len(results)}")

    if not results:
        logger.warning("No results obtained from task processing.")
        return None

    df_results = pd.DataFrame(results)
    df_results["scored_value"] = df_results.apply(apply_reverse_score, axis=1)
    df_results.to_csv(UNIFIED_RESPONSES_CSV_PATH, index=False)
    logger.info(f"Saved responses to {UNIFIED_RESPONSES_CSV_PATH}")

    logger.info("\nProcessing Summary:")
    logger.info(f"Total responses: {len(df_results)}")
    # cost_tracker is updated within api_clients.py, access it directly
    logger.info(f"Token usage by model: {dict(cost_tracker)}") 
    resp_rate = df_results['numeric_score'].notna().sum() / len(df_results) * 100
    logger.info(f"Response rate: {resp_rate:.1f}%")
    return df_results

def run_detailed_analysis(df_results):
    if df_results is None or df_results.empty:
        logger.warning("Skipping detailed analysis as there are no results.")
        return

    logger.info("Running refusal analysis...")
    save_refusal_responses(df_results, output_file="refusal_responses.csv") # Saves to its own file as per notebook

    if "MFQ" in SCALES_TO_RUN:
        logger.info("Running MFQ analysis...")
        calculate_mfq_scores(df_results, output_csv_path='mfq_foundation_scores.csv')
    
    if "RWA" in SCALES_TO_RUN:
        logger.info("Running RWA analysis...")
        rwa_results_df = calculate_rwa_scores(df_results, output_file='rwa_results.csv')
        logger.info("Running RWA average score analysis...")
        calculate_average_scores(df_results, "RWA", 34, output_file_suffix='_avg_results.csv')

    if "LWA" in SCALES_TO_RUN:
        logger.info("Running LWA analysis...")
        lwa_results_df = calculate_lwa_scores(df_results, output_file='lwa_results.csv')
        logger.info("Running LWA average score analysis...")
        calculate_average_scores(df_results, "LWA", 39, output_file_suffix='_avg_results.csv')

def run_ccr_analysis(all_responses_df_for_ccr_defs):
    logger.info("Starting CCR Analysis Framework...")
    sbert_model = None
    try:
        logger.info(f"Loading SentenceTransformer model: {SENTENCE_TRANSFORMER_MODEL_NAME}...")
        sbert_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
        logger.info("SBERT Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model: {e}. Method 1 (Item-to-Item) CCR will not be available.")

    # Ensure the base CSV for CCR definitions exists and is readable
    if not os.path.exists(UNIFIED_RESPONSES_CSV_PATH):
        logger.error(f"CCR Analysis requires {UNIFIED_RESPONSES_CSV_PATH}, but it's missing. Run survey first.")
        return
    
    # df_all_responses_for_ccr = pd.read_csv(UNIFIED_RESPONSES_CSV_PATH)
    # if df_all_responses_for_ccr.empty:
    #     logger.error(f"CCR Analysis: {UNIFIED_RESPONSES_CSV_PATH} is empty.")
    #     return
    # Instead of re-reading, use the passed df
    if all_responses_df_for_ccr_defs is None or all_responses_df_for_ccr_defs.empty:
        logger.error(f"CCR Analysis: Input DataFrame for scale definitions is empty or None.")
        return


    ccr_models_to_analyze = MODELS_TO_RUN # Using the main model list for CCR
    ccr_prompt_styles = PROMPT_STYLES_TO_RUN 
    ccr_scales = ["RWA", "LWA"] # Specific scales for CCR as per notebook

    analysis_results_list = []
    ensure_ccr_temp_dir(CCR_TEMP_DIR)

    for model_name_csv in ccr_models_to_analyze:
        model_name_key_for_gt_speech = SPEECH_MODEL_MAPPING.get(model_name_csv, model_name_csv)
        for prompt_style in ccr_prompt_styles:
            logger.info(f"\nCCR Processing: Model={model_name_csv} (Key: {model_name_key_for_gt_speech}), Prompt={prompt_style}")
            current_run_results = {
                'model_name_key': model_name_key_for_gt_speech,
                'csv_model_name': model_name_csv, 
                'prompt_style': prompt_style
            }
            for scale_name in ccr_scales:
                logger.info(f"  CCR Scale: {scale_name}")
                filtered_data = load_and_filter_data_for_ccr(
                    UNIFIED_RESPONSES_CSV_PATH, model_name_csv, prompt_style, scale_name
                )
                if filtered_data.empty:
                    logger.warning(f"    No data for CCR: {model_name_csv}/{prompt_style}/{scale_name}")
                    current_run_results[f"{scale_name}_item_analysis_score"] = np.nan
                    current_run_results[f"{scale_name}_corpus_analysis_score"] = np.nan
                    current_run_results[f"{scale_name}_speech_analysis_score"] = np.nan
                    continue
                
                scale_def = get_scale_definition_for_ccr(all_responses_df_for_ccr_defs, scale_name)
                if not scale_def:
                    logger.warning(f"    No scale definition for CCR: {scale_name}")
                    current_run_results[f"{scale_name}_item_analysis_score"] = np.nan
                    current_run_results[f"{scale_name}_corpus_analysis_score"] = np.nan
                    current_run_results[f"{scale_name}_speech_analysis_score"] = np.nan
                    continue

                # Method 1: Item-to-Item CCR
                if sbert_model:
                    item_results, item_score = run_item_analysis_ccr(sbert_model, filtered_data)
                    current_run_results[f"{scale_name}_item_analysis_score"] = item_score
                    # Item-level correlation (optional, can be extensive)
                    # ... (code from notebook if needed here for item_corr)
                else:
                    current_run_results[f"{scale_name}_item_analysis_score"] = np.nan

                # Method 2: Corpus of Justifications CCR
                corpus_text = " ".join(filtered_data['justification'].astype(str).tolist())
                if corpus_text.strip():
                    corpus_score = run_corpus_or_speech_analysis_ccr(corpus_text, scale_def, "Corpus")
                    current_run_results[f"{scale_name}_corpus_analysis_score"] = corpus_score
                else:
                    current_run_results[f"{scale_name}_corpus_analysis_score"] = np.nan

                # Method 3: Holistic Speech CCR
                speech = HOLISTIC_SPEECHES.get(model_name_key_for_gt_speech, {}).get(prompt_style)
                if speech:
                    speech_score = run_corpus_or_speech_analysis_ccr(speech, scale_def, "Speech")
                    current_run_results[f"{scale_name}_speech_analysis_score"] = speech_score
                else:
                    current_run_results[f"{scale_name}_speech_analysis_score"] = np.nan
            analysis_results_list.append(current_run_results)

    if analysis_results_list:
        results_df_ccr = pd.DataFrame(analysis_results_list)
        overall_gt_df = pd.DataFrame(OVERALL_GT_DATA)
        comparison_final_df = pd.merge(results_df_ccr, overall_gt_df, 
                                       on=["model_name_key", "prompt_style"], how="left")
        logger.info("\n\n--- Overall CCR Scores Summary & Comparison ---")
        display_cols = ['model_name_key', 'prompt_style']
        for scale in ccr_scales:
            display_cols.extend([
                f"{scale}_item_analysis_score", f"{scale}_corpus_analysis_score",
                f"{scale}_speech_analysis_score", f"{scale}_ground_truth"
            ])
        display_cols = [col for col in display_cols if col in comparison_final_df.columns]
        logger.info(comparison_final_df[display_cols].to_string())

        logger.info("\n--- Correlation of Overall CCR vs. Ground Truth ---")
        for scale in ccr_scales:
            for method_suffix in ["item_analysis_score", "corpus_analysis_score", "speech_analysis_score"]:
                ccr_col = f"{scale}_{method_suffix}"
                gt_col = f"{scale}_ground_truth"
                if ccr_col in comparison_final_df.columns and gt_col in comparison_final_df.columns:
                    valid_comp = comparison_final_df[[ccr_col, gt_col]].dropna()
                    valid_comp[ccr_col] = pd.to_numeric(valid_comp[ccr_col], errors='coerce')
                    valid_comp[gt_col] = pd.to_numeric(valid_comp[gt_col], errors='coerce')
                    valid_comp = valid_comp.dropna()
                    if len(valid_comp) >= 2 and valid_comp[ccr_col].nunique() > 1 and valid_comp[gt_col].nunique() > 1:
                        pearson_r, pearson_p = pearsonr(valid_comp[ccr_col], valid_comp[gt_col])
                        spearman_rho, spearman_p = spearmanr(valid_comp[ccr_col], valid_comp[gt_col])
                        logger.info(f"  Pearson  ({scale} - {method_suffix}): r={pearson_r:.4f}, p={pearson_p:.4f} (N={len(valid_comp)})")
                        logger.info(f"  Spearman ({scale} - {method_suffix}): rho={spearman_rho:.4f}, p={spearman_p:.4f} (N={len(valid_comp)})")
                    else:
                        logger.warning(f"  Cannot compute CCR correlation for {ccr_col} vs {gt_col} (N={len(valid_comp)}). Check variance/data.")
    else:
        logger.info("No CCR analysis results generated.")

def run_plotting():
    logger.info("Generating plots...")
    # Ensure data files for plotting exist
    plot_authoritarian_scores_exists = os.path.exists('rwa_results.csv') and os.path.exists('lwa_results.csv')
    plot_avg_authoritarian_scores_exists = os.path.exists('rwa_avg_results.csv') and os.path.exists('lwa_avg_results.csv')

    if plot_authoritarian_scores_exists:
        logger.info("Plotting normalized authoritarian scores...")
        plot_authoritarian_scores(rwa_csv="rwa_results.csv", lwa_csv="lwa_results.csv")
    else:
        logger.warning("Skipping normalized authoritarian scores plot. Required CSV files missing.")

    if plot_avg_authoritarian_scores_exists:
        logger.info("Plotting average authoritarian scores...")
        plot_avg_authoritarian_scores(rwa_csv="rwa_avg_results.csv", lwa_csv="lwa_avg_results.csv")
    else:
        logger.warning("Skipping average authoritarian scores plot. Required CSV files missing.")
    
    if plot_authoritarian_scores_exists: # Uses same CSVs as the first plot
        logger.info("Plotting separate total authoritarian scores...")
        plot_separate_total_authoritarian_scores(rwa_csv="rwa_results.csv", lwa_csv="lwa_results.csv")
    else:
        logger.warning("Skipping separate total authoritarian scores plot. Required CSV files missing.")

async def main():
    load_dotenv() # Load API keys from .env
    logger.info("Starting main application process...")

    # --- Part 1: Run Surveys and Initial Processing ---
    df_survey_results = await run_survey_and_initial_processing()

    # --- Part 2: Run Detailed Scale-Specific Analyses ---
    if df_survey_results is not None and not df_survey_results.empty:
        run_detailed_analysis(df_survey_results)
    else:
        logger.warning("Skipping detailed analysis due to no survey results.")

    # --- Part 3: Run CCR Analysis (using the initially processed full results for definitions) ---
    # Pass df_survey_results for CCR to use for scale definitions if available
    # Otherwise, CCR analysis part will try to load from UNIFIED_RESPONSES_CSV_PATH
    run_ccr_analysis(df_survey_results) 

    # --- Part 4: Generate Plots ---
    # Plotting functions assume CSV files have been generated by the analysis steps
    run_plotting()

    logger.info("Application finished.")

if __name__ == "__main__":
    asyncio.run(main()) 