import pandas as pd
import logging

logger = logging.getLogger(__name__)

def apply_reverse_score(row):
    """
    Apply reverse scoring for different scales.
    - MFQ scale is 1-5; reversed item => 6 - original
    - LWA and RWA scale is 1-7; reversed item => 8 - original
    """
    score = row["numeric_score"]
    reverse_flag = row.get("reverse_score", False)
    scale_name = row.get("scale_name", "")

    if pd.isna(score):
        return score  # No change if score is NaN

    if not reverse_flag:
        return score  # Return as-is if not a reverse-scored item

    # Handle each scale's reversing logic
    if scale_name == "MFQ":
        return 6 - score
    elif scale_name == "RWA":
        return 8 - score
    elif scale_name == "LWA":
        return 8 - score
    else:
        # If any future scale needs reversing, define it here
        return score

def save_refusal_responses(df, output_file="refusal_responses.csv"):
    """
    After generating unified_responses.csv, call this function to save 
    any rows where the model refused or failed to provide a valid numeric answer.

    Specifically, we look for:
      - The 'justification' that indicates a parser warning or API error.
      - The 'raw_response' text where we logged: 'No valid number found in: ...'

    Adjust filters as needed for your exact logging conventions.
    """
    # Filter rows based on how your warnings/errors are recorded
    df_refusals = df[
        df["justification"].str.contains("PARSER WARNING", na=False)
        | df["justification"].str.contains("API ERROR", na=False)
        | df["raw_response"].str.contains("No valid number found in:", na=False)
    ].copy()

    # Choose the columns that best help you analyze the refusal
    columns_to_save = [
        "model_name",
        "prompt_style",
        "question_id",
        "question_text",
        "justification",
        "raw_response"
    ]
    # Only keep columns actually present
    columns_to_save = [col for col in columns_to_save if col in df_refusals.columns]

    df_refusals = df_refusals[columns_to_save]
    df_refusals.to_csv(output_file, index=False)
    logger.info(f"Refusal responses saved to {output_file}")


MFQ_FOUNDATIONS = {
    'care': [f'MFQ_{i}' for i in [1, 7, 13, 19, 25, 31]],
    'equality': [f'MFQ_{i}' for i in [2, 8, 14, 20, 26, 32]],
    'proportionality': [f'MFQ_{i}' for i in [3, 9, 15, 21, 27, 33]],
    'loyalty': [f'MFQ_{i}' for i in [4, 10, 16, 22, 28, 34]],
    'authority': [f'MFQ_{i}' for i in [5, 11, 17, 23, 29, 35]],
    'purity': [f'MFQ_{i}' for i in [6, 12, 18, 24, 30, 36]]
}

def calculate_mfq_scores(csv_path="unified_responses.csv"):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for MFQ questions only
    df = df[df['scale_name'] == 'MFQ']
    
    # First, average scores across runs for each unique combination
    # of model, prompt, and question
    avg_by_question = df.groupby([
        'model_name',
        'prompt_style',
        'question_id'
    ])['numeric_score'].mean().reset_index()
    
    results = []
    
    # Process each model/prompt combination
    for model in avg_by_question['model_name'].unique():
        for prompt_style in avg_by_question['prompt_style'].unique():
            row = {'model_name': model, 'prompt_style': prompt_style}
            
            # Calculate score for each foundation
            for foundation, questions in MFQ_FOUNDATIONS.items():
                mask = (avg_by_question['model_name'] == model) & \
                      (avg_by_question['prompt_style'] == prompt_style) & \
                      (avg_by_question['question_id'].isin(questions))
                
                foundation_scores = avg_by_question[mask]['numeric_score']
                
                if len(foundation_scores) > 0:
                    # Calculate foundation metrics
                    row[f'{foundation}_mean'] = round(foundation_scores.mean(), 2)
                    row[f'{foundation}_std'] = round(foundation_scores.std(), 2)
                    row[f'{foundation}_count'] = len(foundation_scores)
                    # Add individual question scores for verification
                    for q_id in questions:
                        q_score = avg_by_question[
                            (avg_by_question['model_name'] == model) & 
                            (avg_by_question['prompt_style'] == prompt_style) & 
                            (avg_by_question['question_id'] == q_id)
                        ]['numeric_score'].values
                        if len(q_score) > 0:
                            row[f'{q_id}_score'] = round(q_score[0], 2)
                        else:
                            row[f'{q_id}_score'] = None
                else:
                    row[f'{foundation}_mean'] = None
                    row[f'{foundation}_std'] = None
                    row[f'{foundation}_count'] = 0
                    for q_id in questions:
                        row[f'{q_id}_score'] = None
            
            results.append(row)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['model_name', 'prompt_style'])
    
    results_df.to_csv('mfq_foundation_scores.csv', index=False)
    logger.info("\nMFQ Foundation Scores:")
    logger.info(results_df.to_string())
    
    return results_df

def calculate_rwa_scores(df):
    """
    Calculate both raw and normalized RWA scores for each (model_name, prompt_style).
    Now assumes RWA items are on a 1..7 scale.
    """
    df_rwa = df[df["scale_name"] == "RWA"].copy()
    df_rwa["converted_score"] = df_rwa["scored_value"]
    sum_per_run = df_rwa.groupby(
        ["model_name", "prompt_style", "run_number"], as_index=False
    )["converted_score"].sum()
    min_sum = 34
    max_sum = 34 * 7
    sum_per_run["normalized_score"] = (
        sum_per_run["converted_score"] - min_sum
    ) / (max_sum - min_sum)
    grouped_scores = sum_per_run.groupby(
        ["model_name", "prompt_style"], as_index=False
    ).agg({
        "converted_score": [
            ("rwa_total_mean", "mean"),
            ("rwa_total_std", "std"),
            ("run_count", "count")
        ],
        "normalized_score": [
            ("rwa_norm_mean", "mean"),
            ("rwa_norm_std", "std")
        ]
    }).round(3)
    grouped_scores.columns = [
        "model_name", "prompt_style",
        "rwa_total_mean", "rwa_total_std", "run_count",
        "rwa_norm_mean", "rwa_norm_std"
    ]
    return grouped_scores

def save_rwa_results(grouped_scores, output_file='rwa_results.csv'):
    grouped_scores.to_csv(output_file, index=False)
    logger.info(f"RWA results saved to {output_file}")

def calculate_lwa_scores(df):
    """
    Calculate both raw and normalized LWA scores for each (model_name, prompt_style).
    Now assumes LWA items are on a 1..7 scale.
    """
    df_lwa = df[df["scale_name"] == "LWA"].copy()
    df_lwa["converted_score"] = df_lwa["scored_value"]
    sum_per_run = df_lwa.groupby(
        ["model_name", "prompt_style", "run_number"], as_index=False
    )["converted_score"].sum()
    min_sum = 39
    max_sum = 39 * 7
    sum_per_run["normalized_score"] = (
        sum_per_run["converted_score"] - min_sum
    ) / (max_sum - min_sum)
    grouped_scores = sum_per_run.groupby(
        ["model_name", "prompt_style"], as_index=False
    ).agg({
        "converted_score": [
            ("lwa_total_mean", "mean"),
            ("lwa_total_std", "std"),
            ("run_count", "count")
        ],
        "normalized_score": [
            ("lwa_norm_mean", "mean"),
            ("lwa_norm_std", "std")
        ]
    }).round(3)
    grouped_scores.columns = [
        "model_name", "prompt_style",
        "lwa_total_mean", "lwa_total_std", "run_count",
        "lwa_norm_mean", "lwa_norm_std"
    ]
    return grouped_scores

def save_lwa_results(grouped_scores, output_file='lwa_results.csv'):
    grouped_scores.to_csv(output_file, index=False)
    logger.info(f"LWA results saved to {output_file}")

def calculate_average_scores(df, scale_name, num_questions):
    """
    Compute the average score per question for each (model_name, prompt_style) pair.
    
    Instead of summing all items, we divide the total by the number of questions in the scale.
    """
    df_scale = df[df["scale_name"] == scale_name].copy()
    df_scale["converted_score"] = df_scale["scored_value"]  
    sum_per_run = df_scale.groupby(
        ["model_name", "prompt_style", "run_number"], as_index=False
    )["converted_score"].sum()
    sum_per_run["avg_score_per_question"] = sum_per_run["converted_score"] / num_questions
    grouped_scores = sum_per_run.groupby(
        ["model_name", "prompt_style"], as_index=False
    ).agg({
        "avg_score_per_question": [
            ("avg_total_mean", "mean"),
            ("avg_total_std", "std"),
            ("run_count", "count")
        ]
    }).round(3)
    grouped_scores.columns = [
        "model_name", "prompt_style", "avg_total_mean", "avg_total_std", "run_count"
    ]
    return grouped_scores
