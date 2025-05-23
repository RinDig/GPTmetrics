import pandas as pd

def calculate_average_scores(df, scale_name, num_questions, output_file_suffix='_avg_results.csv'):
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
    
    output_filename = f"{scale_name.lower()}{output_file_suffix}"
    grouped_scores.to_csv(output_filename, index=False)
    print(f"{scale_name} average scores saved to {output_filename}")
    return grouped_scores 