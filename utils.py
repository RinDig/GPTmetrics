import pandas as pd
import os

def load_and_filter_data_for_ccr(csv_path, model_name_filter_csv, prompt_style_filter, scale_name_filter=None):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return pd.DataFrame()

    for col in ['justification', 'question_text']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df_filtered = df[
        (df['model_name'] == model_name_filter_csv) &
        (df['prompt_style'] == prompt_style_filter)
    ]
    if scale_name_filter:
        df_filtered = df_filtered[df_filtered['scale_name'] == scale_name_filter]
    
    if df_filtered.empty:
        print(f"Warning: No data for CCR for CSV model='{model_name_filter_csv}', prompt='{prompt_style_filter}'" +
            (f", scale='{scale_name_filter}'" if scale_name_filter else ""))
    return df_filtered.copy()

def get_scale_definition_for_ccr(df_all_responses, scale_name_filter):
    scale_df = df_all_responses[df_all_responses['scale_name'] == scale_name_filter].copy()
    if scale_df.empty:
        print(f"Warning: No questions for CCR for scale '{scale_name_filter}'.")
        return []
    
    if 'reverse_score' in scale_df.columns:
        scale_df['reverse_score'] = scale_df['reverse_score'].replace({'TRUE': True, 'FALSE': False, 'True': True, 'False': False})
        scale_df['reverse_score'] = scale_df['reverse_score'].astype(bool)
    else:
        print(f"Warning: 'reverse_score' not in CCR scale '{scale_name_filter}'. Assuming False.")
        scale_df['reverse_score'] = False

    scale_df['question_text_cleaned'] = scale_df['question_text'].astype(str).str.strip().str.replace(r'\*$', '', regex=True).str.strip()
    
    if 'question_id' in scale_df.columns:
        scale_df = scale_df.sort_values(by='question_id')
        unique_questions = scale_df.drop_duplicates(subset=['question_id'])
        scale_definition = unique_questions[['question_id', 'question_text_cleaned', 'reverse_score']].rename(
            columns={'question_text_cleaned': 'text'}
        ).to_dict('records')
    else:
        print("Warning: 'question_id' not found in CCR. Using 'question_text' for uniqueness.")
        unique_questions = scale_df.drop_duplicates(subset=['question_text_cleaned'])
        scale_definition = unique_questions[['question_text_cleaned', 'reverse_score']].rename(
            columns={'question_text_cleaned': 'text'}
        ).to_dict('records')
        for i, item in enumerate(scale_definition):
            item['question_id'] = f"item_{i+1}"
            
    if not scale_definition:
        print(f"Warning: CCR scale definition for '{scale_name_filter}' is empty.")
    return scale_definition

def ensure_ccr_temp_dir(temp_dir_path):
    if not os.path.exists(temp_dir_path):
        os.makedirs(temp_dir_path)
        print(f"Created temporary directory for CCR: {temp_dir_path}") 