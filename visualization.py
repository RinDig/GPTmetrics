import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_authoritarian_scores(rwa_csv="rwa_results.csv", lwa_csv="lwa_results.csv"):
    """
    Reads 'rwa_results.csv' and 'lwa_results.csv', merges them,
    and plots both RWA and LWA in a single multi-panel plot.
    """

    # 1. Read in the data
    rwa_df = pd.read_csv(rwa_csv)
    lwa_df = pd.read_csv(lwa_csv)

    # 2. Add a 'scale' column so we know which is which
    rwa_df["scale"] = "RWA"
    lwa_df["scale"] = "LWA"

    # 3. For easier merging, rename columns so they match
    #    We'll keep: model_name, prompt_style, total_mean, total_std, norm_mean, norm_std, run_count
    rwa_df_rename = rwa_df.rename(columns={
        "rwa_total_mean": "total_mean",
        "rwa_total_std": "total_std",
        "rwa_norm_mean": "norm_mean",
        "rwa_norm_std": "norm_std"
    })

    lwa_df_rename = lwa_df.rename(columns={
        "lwa_total_mean": "total_mean",
        "lwa_total_std": "total_std",
        "lwa_norm_mean": "norm_mean",
        "lwa_norm_std": "norm_std"
    })

    # 4. Combine into a single dataframe
    combined_df = pd.concat([rwa_df_rename, lwa_df_rename], ignore_index=True)

    # 5. Define an order for prompt styles
    prompt_order = [
        "extreme_republican",
        "mid_republican",
        "minimal",   # or "neutral"
        "mid_liberal",
        "extreme_liberal"
    ]
    combined_df["prompt_style"] = pd.Categorical(
        combined_df["prompt_style"],
        categories=prompt_order,
        ordered=True
    )

    # 6. Decide what you're plotting on the y-axis:
    #    - "total_mean" (raw sum in 1..9 range)
    #    - "norm_mean"  (normalized 0..1 range)
    y_var = "norm_mean"
    y_label = "Normalized Score (0..1)" if y_var == "norm_mean" else "Total Score"

    # We'll pivot the standard deviation column to match the same selection as y_var.
    combined_df["y_std"] = combined_df.apply(
        lambda row: row["norm_std"] if y_var == "norm_mean" else row["total_std"],
        axis=1
    )

    # Use a pointplot for better readability
    g = sns.catplot(
        data=combined_df,
        x="prompt_style",
        y=y_var,
        hue="model_name",
        col="scale",
        kind="point",
        dodge=True,
        join=True,
        errorbar=None,  # Use manual error bars
        height=5,
        aspect=1.2
    )

    # 8. Add custom error bars
    axes = g.axes.flatten()
    for ax_i, ax in enumerate(axes):
        scale_label = ["RWA", "LWA"][ax_i]  # Because we used col="scale"
        # Filter data for that panel
        panel_data = combined_df[combined_df["scale"] == scale_label]
        
        for model_i, (model_name, subdf) in enumerate(panel_data.groupby("model_name")):
            subdf = subdf.sort_values("prompt_style")
            x_vals = range(len(subdf))
            hue_levels = panel_data["model_name"].unique()
            n_hues = len(hue_levels)
            offset = (model_i - (n_hues-1)/2) * 0.4 / (n_hues-1 if n_hues>1 else 1)

            y_vals = subdf[y_var].values
            y_err = subdf["y_std"].values

            ax.errorbar(
                x=[x + offset for x in x_vals],
                y=y_vals,
                yerr=y_err,
                fmt="none",
                capsize=4,
                color="black",
                alpha=0.8,
                linewidth=1
            )

    # 9. Label the axes
    g.set_axis_labels("Prompt Style", y_label)
    g.set_titles("{col_name} Scores")

    # 10. Adjust legend title, layout
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

    g._legend.set_title("Model")
    plt.tight_layout()
    plt.savefig("authoritarian_scores_plot.png") # Save the plot
    plt.show()


def plot_avg_authoritarian_scores(rwa_csv="rwa_avg_results.csv", lwa_csv="lwa_avg_results.csv"):
    """
    Reads the computed 'rwa_avg_results.csv' and 'lwa_avg_results.csv' 
    and plots the average score per question for RWA and LWA.
    """

    # 1. Load data
    rwa_df = pd.read_csv(rwa_csv)
    lwa_df = pd.read_csv(lwa_csv)

    # 2. Label scales
    rwa_df["scale"] = "RWA"
    lwa_df["scale"] = "LWA"

    # 3. Merge datasets (assuming column names for scores are already 'avg_total_mean' and 'avg_total_std')
    combined_df = pd.concat([rwa_df, lwa_df], ignore_index=True)

    # 4. Define prompt order
    prompt_order = [
        "extreme_republican",
        "mid_republican",
        "minimal",  # or "neutral"
        "mid_liberal",
        "extreme_liberal"
    ]
    combined_df["prompt_style"] = pd.Categorical(
        combined_df["prompt_style"],
        categories=prompt_order,
        ordered=True
    )

    # 5. Plot average score per question for both scales
    g = sns.catplot(
        data=combined_df,
        x="prompt_style",
        y="avg_total_mean",
        hue="model_name",
        col="scale",
        kind="point",
        dodge=True,
        join=True,
        errorbar=None,  # Use manual error bars
        height=5,
        aspect=1.2
    )

    # 6. Add custom error bars
    axes = g.axes.flatten()
    for ax_i, ax in enumerate(axes):
        scale_label = ["RWA", "LWA"][ax_i]
        panel_data = combined_df[combined_df["scale"] == scale_label]
        
        for model_i, (model_name, subdf) in enumerate(panel_data.groupby("model_name")):
            subdf = subdf.sort_values("prompt_style")
            x_vals = range(len(subdf))
            hue_levels = panel_data["model_name"].unique()
            n_hues = len(hue_levels)
            offset = (model_i - (n_hues-1)/2) * 0.4 / (n_hues-1 if n_hues>1 else 1)

            y_vals = subdf["avg_total_mean"].values
            y_err = subdf["avg_total_std"].values

            ax.errorbar(
                x=[x + offset for x in x_vals],
                y=y_vals,
                yerr=y_err,
                fmt="none",
                capsize=4,
                color="black",
                alpha=0.8,
                linewidth=1
            )

    # 7. Final adjustments
    g.set_axis_labels("Prompt Style", "Avg Score per Question (1-7)")
    g.set_titles("{col_name} Scores")
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    g._legend.set_title("Model")
    plt.tight_layout()
    plt.savefig("avg_authoritarian_scores_plot.png") # Save the plot
    plt.show()
