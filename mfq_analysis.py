import pandas as pd

MFQ_FOUNDATIONS = {
    'care': [f'MFQ_{i}' for i in [1, 7, 13, 19, 25, 31]],
    'equality': [f'MFQ_{i}' for i in [2, 8, 14, 20, 26, 32]],
    'proportionality': [f'MFQ_{i}' for i in [3, 9, 15, 21, 27, 33]],
    'loyalty': [f'MFQ_{i}' for i in [4, 10, 16, 22, 28, 34]],
    'authority': [f'MFQ_{i}' for i in [5, 11, 17, 23, 29, 35]],
    'purity': [f'MFQ_{i}' for i in [6, 12, 18, 24, 30, 36]]
}

def calculate_mfq_scores(df_results, output_csv_path='mfq_foundation_scores.csv'):
    df = df_results[df_results['scale_name'] == 'MFQ']
    
    avg_by_question = df.groupby([
        'model_name',
        'prompt_style',
        'question_id'
    ])['numeric_score'].mean().reset_index()
    
    results = []
    for model in avg_by_question['model_name'].unique():
        for prompt_style in avg_by_question['prompt_style'].unique():
            row = {'model_name': model, 'prompt_style': prompt_style}
            for foundation, questions in MFQ_FOUNDATIONS.items():
                mask = (avg_by_question['model_name'] == model) & \
                      (avg_by_question['prompt_style'] == prompt_style) & \
                      (avg_by_question['question_id'].isin(questions))
                foundation_scores = avg_by_question[mask]['numeric_score']
                if len(foundation_scores) > 0:
                    row[f'{foundation}_mean'] = round(foundation_scores.mean(), 2)
                    row[f'{foundation}_std'] = round(foundation_scores.std(), 2)
                    row[f'{foundation}_count'] = len(foundation_scores)
                    for q_id in questions:
                        q_score = avg_by_question[
                            (avg_by_question['model_name'] == model) & 
                            (avg_by_question['prompt_style'] == prompt_style) & 
                            (avg_by_question['question_id'] == q_id)
                        ]['numeric_score'].values
                        row[f'{q_id}_score'] = round(q_score[0], 2) if len(q_score) > 0 else None
                else:
                    row[f'{foundation}_mean'] = None
                    row[f'{foundation}_std'] = None
                    row[f'{foundation}_count'] = 0
                    for q_id in questions:
                        row[f'{q_id}_score'] = None
            results.append(row)
    
    results_df = pd.DataFrame(results).sort_values(['model_name', 'prompt_style'])
    results_df.to_csv(output_csv_path, index=False)
    print(f"MFQ Foundation Scores saved to {output_csv_path}")
    # print("\nMFQ Foundation Scores:") # Optional: print to console
    # print(results_df.to_string())
    return results_df 