import streamlit as st
import pandas as pd
import asyncio
import os
import logging
from dotenv import load_dotenv

# Import functions from your existing codebase
from configs.app_config import (
    SCALES_TO_RUN as DEFAULT_SCALES,
    PROMPT_STYLES_TO_RUN as DEFAULT_PROMPT_STYLES,
    MODELS_TO_RUN as DEFAULT_MODELS,
    NUM_CALLS_TEST as DEFAULT_NUM_CALLS,
    UNIFIED_RESPONSES_CSV_PATH,
    HOLISTIC_SPEECHES,
    SPEECH_MODEL_MAPPING,
    OVERALL_GT_DATA,
    CCR_TEMP_DIR,
    SENTENCE_TRANSFORMER_MODEL_NAME
)
from scales.scale_definitions import ALL_QUESTIONS # For display or selection if needed
from data_processing.task_builder import build_tasks
from utils.task_processor import process_tasks_in_chunks
from analysis.scoring import apply_reverse_score
from analysis.refusal_analysis import save_refusal_responses
from analysis.mfq_analysis import calculate_mfq_scores
from analysis.rwa_analysis import calculate_rwa_scores
from analysis.lwa_analysis import calculate_lwa_scores
from analysis.average_score_analysis import calculate_average_scores
from plotting.authoritarian_plots import (
    plot_authoritarian_scores,
    plot_avg_authoritarian_scores,
    plot_separate_total_authoritarian_scores
)
from analysis.ccr.utils import (
    load_and_filter_data_for_ccr, 
    get_scale_definition_for_ccr,
    ensure_ccr_temp_dir
)
from analysis.ccr.analyzer import (
    run_item_analysis_ccr, 
    run_corpus_or_speech_analysis_ccr
)
from llm_interface.api_clients import cost_tracker # Import cost_tracker

# For SBERT model loading in CCR
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, spearmanr
import numpy as np # for np.nan in CCR

# Setup logging for the dashboard
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Helper to run async functions in Streamlit
async def run_async_in_streamlit(func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func(*args))
    loop.close()
    return result 

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="GPT Metrics Dashboard")
st.title("GPT Metrics Analysis Dashboard")

# Initialize session state variables if they don't exist
if 'survey_data_generated' not in st.session_state:
    st.session_state.survey_data_generated = False
if 'df_results' not in st.session_state:
    st.session_state.df_results = None
if 'standard_analysis_done' not in st.session_state:
    st.session_state.standard_analysis_done = False
if 'ccr_analysis_done' not in st.session_state:
    st.session_state.ccr_analysis_done = False

# --- Sidebar for Configuration ---
st.sidebar.header("Pipeline Configuration")

# Get available scales, models, prompt styles for multiselect
available_scales = list(set(q['scale_name'] for q in ALL_QUESTIONS))
available_models = list(DEFAULT_MODELS) # Or from MODEL_CONFIG keys
available_prompt_styles = list(DEFAULT_PROMPT_STYLES) # Or from PROMPT_TEMPLATES keys

# Use a form for configurations to batch selections
with st.sidebar.form("config_form"):
    st.subheader("Survey Generation Settings")
    selected_models = st.multiselect("Select Models to Run:", available_models, default=DEFAULT_MODELS)
    selected_scales = st.multiselect("Select Scales to Run:", available_scales, default=DEFAULT_SCALES)
    selected_prompt_styles = st.multiselect("Select Prompt Styles:", available_prompt_styles, default=DEFAULT_PROMPT_STYLES)
    num_calls = st.number_input("Number of API Calls per Question/Prompt/Model:", min_value=1, value=DEFAULT_NUM_CALLS, step=1)
    
    st.subheader("CCR Analysis Settings (if run)")
    # These are usually fixed but could be made configurable if needed
    # For now, these will use defaults from app_config
    st.text(f"SBERT Model: {SENTENCE_TRANSFORMER_MODEL_NAME}")
    # If you want to select specific models/prompts for CCR different from survey generation:
    ccr_models = st.multiselect("Models for CCR Analysis:", available_models, default=DEFAULT_MODELS, key="ccr_models")
    ccr_prompts = st.multiselect("Prompt Styles for CCR Analysis:", available_prompt_styles, default=DEFAULT_PROMPT_STYLES, key="ccr_prompts")
    ccr_scales_options = ["RWA", "LWA"] # CCR specific scales
    ccr_selected_scales = st.multiselect("Scales for CCR Analysis:", ccr_scales_options, default=ccr_scales_options, key="ccr_scales")

    submitted_config = st.form_submit_button("Apply Configuration")

if submitted_config:
    # Update session state with new configurations
    # These will be read by the main.py functions if we modify them to accept these as args
    # For simplicity here, we are overriding the defaults from app_config by updating them in session_state
    # or by passing them to adapted functions.
    st.session_state.selected_models = selected_models
    st.session_state.selected_scales = selected_scales
    st.session_state.selected_prompt_styles = selected_prompt_styles
    st.session_state.num_calls = num_calls
    st.session_state.ccr_models = ccr_models
    st.session_state.ccr_prompts = ccr_prompts
    st.session_state.ccr_selected_scales = ccr_selected_scales
    st.success("Configuration applied!")

# Display current configuration (use session state if set, else defaults)
st.sidebar.markdown("---_Current Settings (Applied)_---")
st.sidebar.json({
    "Survey Models": st.session_state.get('selected_models', DEFAULT_MODELS),
    "Survey Scales": st.session_state.get('selected_scales', DEFAULT_SCALES),
    "Survey Prompt Styles": st.session_state.get('selected_prompt_styles', DEFAULT_PROMPT_STYLES),
    "API Calls per Task": st.session_state.get('num_calls', DEFAULT_NUM_CALLS),
    "CCR Models": st.session_state.get('ccr_models', DEFAULT_MODELS),
    "CCR Prompts": st.session_state.get('ccr_prompts', DEFAULT_PROMPT_STYLES),
    "CCR Scales": st.session_state.get('ccr_selected_scales', ["RWA", "LWA"])
})

# --- Main Application Area ---
col1, col2, col3 = st.columns(3) 

# --- Stage 1: Generate Survey Data ---
with col1:
    st.header("Stage 1: Generate Survey Data")
    if st.button("Run Survey Data Generation"):
        if not st.session_state.get('selected_models'):
            st.error("Please apply a configuration in the sidebar first.")
        else:
            with st.spinner("Generating survey data... This may take a while."):
                # Adapt build_tasks to use selected configs
                tasks = build_tasks(
                    scales_to_run=st.session_state.selected_scales,
                    models_to_run=st.session_state.selected_models,
                    prompt_styles_to_run=st.session_state.selected_prompt_styles,
                    num_calls_test=st.session_state.num_calls
                )
                if not tasks:
                    st.warning("No tasks were built based on current selections.")
                else:
                    st.info(f"Built {len(tasks)} tasks for survey generation.")
                    # process_tasks_in_chunks already uses MAX_CONCURRENT_CALLS from app_config
                    results = run_async_in_streamlit(process_tasks_in_chunks, tasks)
                    
                    if results:
                        df_results = pd.DataFrame(results)
                        df_results["scored_value"] = df_results.apply(apply_reverse_score, axis=1)
                        df_results.to_csv(UNIFIED_RESPONSES_CSV_PATH, index=False)
                        st.session_state.df_results = df_results
                        st.session_state.survey_data_generated = True
                        st.success(f"Survey data generated and saved to {UNIFIED_RESPONSES_CSV_PATH}")
                        st.info(f"Total responses: {len(df_results)}")
                        st.info(f"Token usage: {dict(cost_tracker)}")
                        with st.expander("View Generated Data Head"):
                            st.dataframe(df_results.head())
                    else:
                        st.error("No results obtained from task processing.")

if st.session_state.survey_data_generated:
    st.success("Stage 1 (Survey Data Generation) Completed.")
    if st.session_state.df_results is not None:
        st.metric("Total Rows in Survey Data", len(st.session_state.df_results))
else:
    st.info("Stage 1: Survey Data Generation not yet run or completed.")

# --- Stage 2: Standard Analysis & Plotting ---
with col2:
    st.header("Stage 2: Analysis & Plotting")
    run_standard_analysis = st.button("Run Standard Analysis & Plotting", disabled=not st.session_state.survey_data_generated)

    if run_standard_analysis:
        if st.session_state.df_results is None or st.session_state.df_results.empty:
            st.error("No survey data (df_results) available. Please run Stage 1 first.")
        else:
            df_results = st.session_state.df_results
            with st.spinner("Running standard analysis and generating plots..."):
                st.write("Running refusal analysis...")
                save_refusal_responses(df_results, output_file="refusal_responses.csv")
                st.info("Refusal responses saved to refusal_responses.csv")

                # We need to pass the selected scales for analysis from session state
                current_scales_to_run = st.session_state.get('selected_scales', DEFAULT_SCALES)

                if "MFQ" in current_scales_to_run:
                    st.write("Running MFQ analysis...")
                    calculate_mfq_scores(df_results, output_csv_path='mfq_foundation_scores.csv')
                    st.info("MFQ scores saved to mfq_foundation_scores.csv")
                
                if "RWA" in current_scales_to_run:
                    st.write("Running RWA analysis...")
                    calculate_rwa_scores(df_results, output_file='rwa_results.csv')
                    st.info("RWA results saved to rwa_results.csv")
                    st.write("Running RWA average score analysis...")
                    calculate_average_scores(df_results, "RWA", 34, output_file_suffix='_avg_results.csv')
                    st.info("RWA average scores saved to rwa_avg_results.csv")

                if "LWA" in current_scales_to_run:
                    st.write("Running LWA analysis...")
                    calculate_lwa_scores(df_results, output_file='lwa_results.csv')
                    st.info("LWA results saved to lwa_results.csv")
                    st.write("Running LWA average score analysis...")
                    calculate_average_scores(df_results, "LWA", 39, output_file_suffix='_avg_results.csv')
                    st.info("LWA average scores saved to lwa_avg_results.csv")

                st.write("Generating plots...")
                # Adapt plotting functions to return fig and display with st.pyplot
                # For simplicity, we'll call them and they'll use plt.show() for now, 
                # which Streamlit can sometimes capture. Proper way is to pass fig.
                
                plot_files_exist_main = os.path.exists('rwa_results.csv') and os.path.exists('lwa_results.csv')
                plot_files_exist_avg = os.path.exists('rwa_avg_results.csv') and os.path.exists('lwa_avg_results.csv')

                if plot_files_exist_main:
                    st.subheader("Normalized Authoritarian Scores")
                    fig1 = plot_authoritarian_scores(rwa_csv="rwa_results.csv", lwa_csv="lwa_results.csv", show_plots=False)
                    if fig1: st.pyplot(fig1)
                    else: st.info("Plot 1 could not be generated.")
                    
                    st.subheader("Separate Total Authoritarian Scores")
                    # This function creates multiple plots, need to handle it
                    # For now, it will show them sequentially if plt.show() is used internally
                    # Or adapt it to return a list of figures.
                    plot_separate_total_authoritarian_scores(rwa_csv="rwa_results.csv", lwa_csv="lwa_results.csv") # Uses plt.show()
                else:
                    st.warning("CSV files for some plots are missing (rwa_results.csv, lwa_results.csv)")

                if plot_files_exist_avg:
                    st.subheader("Average Authoritarian Scores per Question")
                    fig2 = plot_avg_authoritarian_scores(rwa_csv="rwa_avg_results.csv", lwa_csv="lwa_avg_results.csv", show_plots=False)
                    if fig2: st.pyplot(fig2)
                    else: st.info("Plot 2 could not be generated.")
                else:
                     st.warning("CSV files for average score plots are missing (rwa_avg_results.csv, lwa_avg_results.csv)")

                st.session_state.standard_analysis_done = True
                st.success("Standard Analysis and Plotting Completed!")

if st.session_state.standard_analysis_done:
    st.success("Stage 2 (Standard Analysis & Plotting) Completed.")
else:
    st.info("Stage 2: Standard Analysis & Plotting not yet run or completed.")

# --- Stage 3: CCR Analysis ---
with col3:
    st.header("Stage 3: CCR Analysis")
    run_ccr = st.button("Run CCR Analysis", disabled=not st.session_state.survey_data_generated)

    if run_ccr:
        if st.session_state.df_results is None or st.session_state.df_results.empty:
            st.error("No survey data (df_results) available. Please run Stage 1 first for CCR definitions.")
        else:
            df_all_responses_for_ccr_defs = st.session_state.df_results
            with st.spinner("Running CCR Analysis..."):
                sbert_model_instance = None
                try:
                    st.write(f"Loading SBERT model: {SENTENCE_TRANSFORMER_MODEL_NAME}")
                    sbert_model_instance = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
                    st.info("SBERT Model loaded.")
                except Exception as e_sbert:
                    st.error(f"Error loading SBERT model: {e_sbert}")
                
                ensure_ccr_temp_dir(CCR_TEMP_DIR) # Make sure temp dir exists
                ccr_analysis_results_list = []

                # Use CCR specific configurations from session state
                ccr_models_to_run = st.session_state.get('ccr_models', DEFAULT_MODELS)
                ccr_prompt_styles_to_run = st.session_state.get('ccr_prompts', DEFAULT_PROMPT_STYLES)
                ccr_scales_to_analyze = st.session_state.get('ccr_selected_scales', ["RWA", "LWA"])

                for model_name_csv_ccr in ccr_models_to_run:
                    model_name_key_ccr = SPEECH_MODEL_MAPPING.get(model_name_csv_ccr, model_name_csv_ccr)
                    for prompt_style_ccr in ccr_prompt_styles_to_run:
                        st.write(f"CCR Processing: Model={model_name_csv_ccr}, Prompt={prompt_style_ccr}")
                        current_run_res_ccr = {
                            'model_name_key': model_name_key_ccr,
                            'csv_model_name': model_name_csv_ccr,
                            'prompt_style': prompt_style_ccr
                        }
                        for scale_name_ccr in ccr_scales_to_analyze:
                            st.write(f"  CCR Scale: {scale_name_ccr}")
                            filtered_data_ccr = load_and_filter_data_for_ccr(
                                UNIFIED_RESPONSES_CSV_PATH, model_name_csv_ccr, prompt_style_ccr, scale_name_ccr
                            )
                            if filtered_data_ccr.empty:
                                st.warning(f"    No data for CCR: {model_name_csv_ccr}/{prompt_style_ccr}/{scale_name_ccr}")
                                for suffix in ["item_analysis_score", "corpus_analysis_score", "speech_analysis_score"]:
                                    current_run_res_ccr[f"{scale_name_ccr}_{suffix}"] = np.nan
                                continue
                            
                            scale_def_ccr = get_scale_definition_for_ccr(df_all_responses_for_ccr_defs, scale_name_ccr)
                            if not scale_def_ccr:
                                st.warning(f"    No scale definition for CCR: {scale_name_ccr}")
                                for suffix in ["item_analysis_score", "corpus_analysis_score", "speech_analysis_score"]:
                                    current_run_res_ccr[f"{scale_name_ccr}_{suffix}"] = np.nan
                                continue

                            if sbert_model_instance:
                                _, item_score_ccr = run_item_analysis_ccr(sbert_model_instance, filtered_data_ccr)
                                current_run_res_ccr[f"{scale_name_ccr}_item_analysis_score"] = item_score_ccr
                            else:
                                current_run_res_ccr[f"{scale_name_ccr}_item_analysis_score"] = np.nan

                            corpus_text_ccr = " ".join(filtered_data_ccr['justification'].astype(str).tolist())
                            if corpus_text_ccr.strip():
                                corpus_score_ccr = run_corpus_or_speech_analysis_ccr(corpus_text_ccr, scale_def_ccr, "Corpus")
                                current_run_res_ccr[f"{scale_name_ccr}_corpus_analysis_score"] = corpus_score_ccr
                            else:
                                current_run_res_ccr[f"{scale_name_ccr}_corpus_analysis_score"] = np.nan
                            
                            speech_ccr = HOLISTIC_SPEECHES.get(model_name_key_ccr, {}).get(prompt_style_ccr)
                            if speech_ccr:
                                speech_score_ccr = run_corpus_or_speech_analysis_ccr(speech_ccr, scale_def_ccr, "Speech")
                                current_run_res_ccr[f"{scale_name_ccr}_speech_analysis_score"] = speech_score_ccr
                            else:
                                current_run_res_ccr[f"{scale_name_ccr}_speech_analysis_score"] = np.nan
                        ccr_analysis_results_list.append(current_run_res_ccr)
                
                if ccr_analysis_results_list:
                    results_df_ccr_final = pd.DataFrame(ccr_analysis_results_list)
                    overall_gt_df_ccr = pd.DataFrame(OVERALL_GT_DATA)
                    comparison_df_ccr = pd.merge(results_df_ccr_final, overall_gt_df_ccr, 
                                               on=["model_name_key", "prompt_style"], how="left")
                    st.subheader("CCR Scores Summary & Comparison")
                    display_cols_ccr = ['model_name_key', 'prompt_style'] + \
                                       [f"{s}_{m}_score" for s in ccr_scales_to_analyze 
                                        for m in ["item_analysis", "corpus_analysis", "speech_analysis"]] + \
                                       [f"{s}_ground_truth" for s in ccr_scales_to_analyze]
                    display_cols_ccr_filtered = [col for col in display_cols_ccr if col in comparison_df_ccr.columns]
                    st.dataframe(comparison_df_ccr[display_cols_ccr_filtered])

                    st.subheader("CCR Correlation vs. Ground Truth")
                    for scale_corr in ccr_scales_to_analyze:
                        for method_suffix_corr in ["item_analysis_score", "corpus_analysis_score", "speech_analysis_score"]:
                            ccr_col_corr = f"{scale_corr}_{method_suffix_corr}"
                            gt_col_corr = f"{scale_corr}_ground_truth"
                            if ccr_col_corr in comparison_df_ccr.columns and gt_col_corr in comparison_df_ccr.columns:
                                valid_comp_corr = comparison_df_ccr[[ccr_col_corr, gt_col_corr]].dropna()
                                valid_comp_corr[ccr_col_corr] = pd.to_numeric(valid_comp_corr[ccr_col_corr], errors='coerce')
                                valid_comp_corr[gt_col_corr] = pd.to_numeric(valid_comp_corr[gt_col_corr], errors='coerce')
                                valid_comp_corr.dropna(inplace=True)
                                if len(valid_comp_corr) >= 2 and valid_comp_corr[ccr_col_corr].nunique() > 1 and valid_comp_corr[gt_col_corr].nunique() > 1:
                                    pearson_r_c, pearson_p_c = pearsonr(valid_comp_corr[ccr_col_corr], valid_comp_corr[gt_col_corr])
                                    spearman_rho_c, spearman_p_c = spearmanr(valid_comp_corr[ccr_col_corr], valid_comp_corr[gt_col_corr])
                                    st.text(f"  Pearson  ({scale_corr} - {method_suffix_corr}): r={pearson_r_c:.4f}, p={pearson_p_c:.4f} (N={len(valid_comp_corr)})")
                                    st.text(f"  Spearman ({scale_corr} - {method_suffix_corr}): rho={spearman_rho_c:.4f}, p={spearman_p_c:.4f} (N={len(valid_comp_corr)})")
                                else:
                                    st.text(f"  Cannot compute CCR correlation for {ccr_col_corr} vs {gt_col_corr} (N={len(valid_comp_corr)}). Check variance/data.")
                st.session_state.ccr_analysis_done = True
                st.success("CCR Analysis Completed!")

if st.session_state.ccr_analysis_done:
    st.success("Stage 3 (CCR Analysis) Completed.")
else:
    st.info("Stage 3: CCR Analysis not yet run or completed.")

# Instructions to run:
# 1. Save this as dashboard.py in gpt_metrics_project folder.
# 2. Make sure all other modules (configs, data_processing, etc.) are in place.
# 3. Create your .env file with API keys in gpt_metrics_project.
# 4. Run in terminal: streamlit run gpt_metrics_project/dashboard.py 