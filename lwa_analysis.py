import pandas as pd

def calculate_lwa_scores(df, output_file='lwa_results.csv'):
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
    
    grouped_scores.to_csv(output_file, index=False)
    print(f"LWA results saved to {output_file}")
    # print("\nLWA Scores by Model and Prompt:") # Optional
    # print(grouped_scores)
    return grouped_scores