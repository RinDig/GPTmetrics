import pandas as pd
import numpy as np
import os
from sentence_transformers import util
from ccr import ccr_wrapper # Assuming ccr.py or ccr package is in PYTHONPATH or installed

from .utils import ensure_ccr_temp_dir
from configs.app_config import CCR_TEMP_DIR

def run_item_analysis_ccr(sbert_model, filtered_df_for_scale):
    print(f"  Running Item-to-Item CCR Analysis...")
    if filtered_df_for_scale.empty:
        print("    No data to analyze for item-to-item CCR.")
        return [], np.nan

    # Check for required columns
    required_cols = ['justification', 'question_text', 'reverse_score']
    if not all(col in filtered_df_for_scale.columns for col in required_cols):
        print(f"    Error: Required columns missing. Need: {required_cols}")
        return [], np.nan
        
    filtered_df_for_scale['justification'] = filtered_df_for_scale['justification'].astype(str)
    filtered_df_for_scale['question_text_original_row'] = filtered_df_for_scale['question_text'].astype(str)

    texts_to_embed_questions = filtered_df_for_scale['question_text_original_row'].tolist()
    texts_to_embed_justifications = filtered_df_for_scale['justification'].tolist()

    if not texts_to_embed_questions or not texts_to_embed_justifications:
        print("    No texts found for SBERT embedding in item-to-item CCR analysis.")
        return [], np.nan

    try:
        question_embeddings = sbert_model.encode(texts_to_embed_questions, convert_to_tensor=True, show_progress_bar=False)
        justification_embeddings = sbert_model.encode(texts_to_embed_justifications, convert_to_tensor=True, show_progress_bar=False)
    except Exception as e:
        print(f"    SBERT encoding error in item-to-item CCR: {e}")
        return [], np.nan

    item_level_results = []
    all_item_ccr_scores_adjusted = []

    for i in range(len(texts_to_embed_questions)):
        similarity_tensor = util.cos_sim(question_embeddings[i], justification_embeddings[i])
        item_ccr_score = similarity_tensor.item()
        
        is_reversed = filtered_df_for_scale.iloc[i]['reverse_score'] 
        if not isinstance(is_reversed, bool):
            is_reversed = str(is_reversed).strip().upper() == 'TRUE'

        adjusted_score = -item_ccr_score if is_reversed else item_ccr_score
        all_item_ccr_scores_adjusted.append(adjusted_score)
        
        item_level_results.append({
            'question_id': filtered_df_for_scale.iloc[i].get('question_id', f'q_{i}'),
            'question_text': texts_to_embed_questions[i],
            'justification': texts_to_embed_justifications[i],
            'item_ccr_raw_similarity': item_ccr_score,
            'item_ccr_adjusted_similarity': adjusted_score,
            'item_ground_truth_score': filtered_df_for_scale.iloc[i].get('numeric_score')
        })
    
    overall_score = np.nanmean(all_item_ccr_scores_adjusted) if all_item_ccr_scores_adjusted else np.nan
    print(f"    Item-to-Item Overall CCR Score: {overall_score:.4f}")
    return item_level_results, overall_score

def run_corpus_or_speech_analysis_ccr(text_to_analyze, scale_definition_list, analysis_type_name):
    print(f"  Running {analysis_type_name} CCR Analysis...")
    ensure_ccr_temp_dir(CCR_TEMP_DIR)

    if not text_to_analyze or not scale_definition_list:
        print(f"    No text or insufficient scale definition for {analysis_type_name} CCR.")
        return np.nan

    temp_data_file = os.path.join(CCR_TEMP_DIR, "temp_single_text_data.csv")
    temp_q_file = os.path.join(CCR_TEMP_DIR, "temp_scale_questions.csv")

    pd.DataFrame([{'text_col': text_to_analyze}]).to_csv(temp_data_file, index=False)
    q_df_data = [{'item_text': item['text']} for item in scale_definition_list]
    pd.DataFrame(q_df_data).to_csv(temp_q_file, index=False)

    overall_score = np.nan
    try:
        raw_ccr_output_df = ccr_wrapper(
            data_file=temp_data_file,
            data_col='text_col',
            q_file=temp_q_file,
            q_col='item_text'
        )
        if not raw_ccr_output_df.empty:
            adjusted_item_scores = []
            for i, scale_item_meta in enumerate(scale_definition_list):
                sim_col_name = f"sim_item_{i+1}"
                if sim_col_name in raw_ccr_output_df.columns:
                    raw_similarity = raw_ccr_output_df.iloc[0][sim_col_name]
                    is_reversed = scale_item_meta['reverse_score']
                    if not isinstance(is_reversed, bool):
                        is_reversed = str(is_reversed).strip().upper() == 'TRUE'
                    adjusted_score = -raw_similarity if is_reversed else raw_similarity
                    adjusted_item_scores.append(adjusted_score)
                else:
                    print(f"    Warning: Sim col {sim_col_name} not found in {analysis_type_name} CCR output.")
                    adjusted_item_scores.append(np.nan)
            overall_score = np.nanmean(adjusted_item_scores) if adjusted_item_scores else np.nan
            print(f"    {analysis_type_name} Overall CCR Score: {overall_score:.4f}")
        else:
            print(f"    Warning: ccr_wrapper returned empty for {analysis_type_name}.")
    except Exception as e:
        print(f"    Error in ccr_wrapper for {analysis_type_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_data_file): os.remove(temp_data_file)
        if os.path.exists(temp_q_file): os.remove(temp_q_file)
    return overall_score 