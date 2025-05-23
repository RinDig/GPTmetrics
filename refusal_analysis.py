import pandas as pd

def save_refusal_responses(df, output_file="refusal_responses.csv"):
    df_refusals = df[
        df["justification"].str.contains("PARSER WARNING", na=False)
        | df["justification"].str.contains("API ERROR", na=False)
        | df["raw_response"].str.contains("No valid number found in:", na=False)
    ].copy()

    columns_to_save = [
        "model_name",
        "prompt_style",
        "question_id",
        "question_text",
        "justification",
        "raw_response"
    ]
    columns_to_save = [col for col in columns_to_save if col in df_refusals.columns]

    df_refusals = df_refusals[columns_to_save]
    df_refusals.to_csv(output_file, index=False)
    print(f"Refusal responses saved to {output_file}") 