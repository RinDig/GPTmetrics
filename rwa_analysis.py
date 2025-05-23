import pandas as pd

def calculate_rwa_scores(df, output_file='rwa_results.csv'):
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
    
    grouped_scores.to_csv(output_file, index=False)
    print(f"RWA results saved to {output_file}")
    # print("\nRWA Scores by Model and Prompt:") # Optional
    # print(grouped_scores)
    return grouped_scores 