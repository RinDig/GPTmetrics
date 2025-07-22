"""Analysis utilities for LLM Survey Pipeline"""
import pandas as pd
from llm_survey_pipeline.config import MFQ_FOUNDATIONS


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
    print(f"Refusal responses saved to {output_file}")


def calculate_mfq_scores(csv_path):
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
    
    # Convert to DataFrame and sort by model name and prompt style
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['model_name', 'prompt_style'])
    
    # Save results
    results_df.to_csv('mfq_foundation_scores.csv', index=False)
    print("\nMFQ Foundation Scores:")
    print(results_df.to_string())
    
    return results_df