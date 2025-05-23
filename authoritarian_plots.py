import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_authoritarian_scores(rwa_csv="rwa_results.csv", lwa_csv="lwa_results.csv"):
    rwa_df = pd.read_csv(rwa_csv)
    lwa_df = pd.read_csv(lwa_csv)

    rwa_df["scale"] = "RWA"
    lwa_df["scale"] = "LWA"

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

    combined_df = pd.concat([rwa_df_rename, lwa_df_rename], ignore_index=True)

    prompt_order = [
        "extreme_republican", "mid_republican", "minimal",
        "mid_liberal", "extreme_liberal"
    ]
    combined_df["prompt_style"] = pd.Categorical(
        combined_df["prompt_style"],
        categories=prompt_order,
        ordered=True
    )

    y_var = "norm_mean"
    y_label = "Normalized Score (0..1)" if y_var == "norm_mean" else "Total Score"

    combined_df["y_std"] = combined_df.apply(
        lambda row: row["norm_std"] if y_var == "norm_mean" else row["total_std"],
        axis=1
    )

    g = sns.catplot(
        data=combined_df,
        x="prompt_style", y=y_var, hue="model_name", col="scale",
        kind="point", dodge=True, join=True, ci=None,
        height=5, aspect=1.2
    )

    axes = g.axes.flatten()
    for ax_i, ax in enumerate(axes):
        scale_label = ["RWA", "LWA"][ax_i]
        panel_data = combined_df[combined_df["scale"] == scale_label]
        
        for model_i, (model_name, subdf) in enumerate(panel_data.groupby("model_name")):
            subdf = subdf.sort_values("prompt_style")
            x_vals = range(len(subdf))
            hue_levels = panel_data["model_name"].unique()
            n_hues = len(hue_levels)
            offset = (model_i - (n_hues - 1) / 2) * 0.4 / (n_hues if n_hues > 1 else 1)

            y_vals = subdf[y_var].values
            y_err = subdf["y_std"].values

            ax.errorbar(
                x=[x + offset for x in x_vals],
                y=y_vals, yerr=y_err,
                fmt="none", capsize=4, color="black", alpha=0.8, linewidth=1
            )

    g.set_axis_labels("Prompt Style", y_label)
    g.set_titles("{col_name} Scores")
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    g._legend.set_title("Model")
    plt.tight_layout()
    plt.show()

def plot_avg_authoritarian_scores(rwa_csv="rwa_avg_results.csv", lwa_csv="lwa_avg_results.csv"):
    rwa_df = pd.read_csv(rwa_csv)
    lwa_df = pd.read_csv(lwa_csv)
    rwa_df["scale"] = "RWA"
    lwa_df["scale"] = "LWA"
    combined_df = pd.concat([rwa_df, lwa_df], ignore_index=True)
    prompt_order = [
        "extreme_republican", "mid_republican", "minimal", 
        "mid_liberal", "extreme_liberal"
    ]
    combined_df["prompt_style"] = pd.Categorical(
        combined_df["prompt_style"], categories=prompt_order, ordered=True
    )
    g = sns.catplot(
        data=combined_df, x="prompt_style", y="avg_total_mean",
        hue="model_name", col="scale", kind="point", dodge=True,
        join=True, ci=None, height=5, aspect=1.2
    )
    axes = g.axes.flatten()
    for ax_i, ax in enumerate(axes):
        scale_label = ["RWA", "LWA"][ax_i]
        panel_data = combined_df[combined_df["scale"] == scale_label]
        for model_i, (model_name, subdf) in enumerate(panel_data.groupby("model_name")):
            subdf = subdf.sort_values("prompt_style")
            x_vals = range(len(subdf))
            hue_levels = panel_data["model_name"].unique()
            n_hues = len(hue_levels)
            offset = (model_i - (n_hues - 1) / 2) * 0.4 / (n_hues if n_hues > 1 else 1)
            y_vals = subdf["avg_total_mean"].values
            y_err = subdf["avg_total_std"].values
            ax.errorbar(
                x=[x + offset for x in x_vals], y=y_vals, yerr=y_err,
                fmt="none", capsize=4, color="black", alpha=0.8, linewidth=1
            )
    g.set_axis_labels("Prompt Style", "Avg Score per Question (1-7)")
    g.set_titles("{col_name} Scores")
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    g._legend.set_title("Model")
    plt.tight_layout()
    plt.show()

def plot_separate_total_authoritarian_scores(rwa_csv="rwa_results.csv", lwa_csv="lwa_results.csv"):
    try:
        rwa_df = pd.read_csv(rwa_csv)
        lwa_df = pd.read_csv(lwa_csv)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the CSV files exist.")
        return

    rwa_df["scale"] = "RWA"
    lwa_df["scale"] = "LWA"

    rwa_df_rename = rwa_df.rename(columns={
        "rwa_total_mean": "total_mean", "rwa_total_std": "total_std",
    })[["model_name", "prompt_style", "total_mean", "total_std", "scale", "run_count"]]

    lwa_df_rename = lwa_df.rename(columns={
        "lwa_total_mean": "total_mean", "lwa_total_std": "total_std",
    })[["model_name", "prompt_style", "total_mean", "total_std", "scale", "run_count"]]

    combined_df = pd.concat([rwa_df_rename, lwa_df_rename], ignore_index=True)

    prompt_order = [
        "extreme_republican", "mid_republican", "minimal",
        "mid_liberal", "extreme_liberal"
    ]
    combined_df = combined_df[combined_df["prompt_style"].isin(prompt_order)]
    combined_df["prompt_style"] = pd.Categorical(
        combined_df["prompt_style"], categories=prompt_order, ordered=True
    )
    combined_df = combined_df.sort_values("prompt_style")

    def create_single_plot(data, scale_name):
        plt.figure(figsize=(8, 5))
        ax = sns.pointplot(
            data=data, x="prompt_style", y="total_mean", hue="model_name",
            dodge=True, errorbar=None, join=True, order=prompt_order
        )
        hue_levels = data["model_name"].unique()
        n_hues = len(hue_levels)
        point_width = 0.4
        x_coords = {}
        for i, prompt in enumerate(prompt_order):
            prompt_data = data[data["prompt_style"] == prompt]
            if prompt_data.empty: continue
            offsets = [(model_idx - (n_hues - 1) / 2) * point_width / (n_hues if n_hues > 1 else 1) for model_idx in range(n_hues)]
            for model_idx, model_name_val in enumerate(hue_levels):
                 if model_name_val not in prompt_data["model_name"].values: continue
                 if model_name_val not in x_coords: x_coords[model_name_val] = {}
                 x_coords[model_name_val][prompt] = i + offsets[model_idx]
        for model_idx, model_name_val in enumerate(hue_levels):
            subdf = data[data["model_name"] == model_name_val].sort_values("prompt_style")
            if subdf.empty: continue
            current_x_vals = [x_coords[model_name_val].get(p, None) for p in subdf["prompt_style"]]
            valid_indices = [idx for idx, x_val in enumerate(current_x_vals) if x_val is not None]
            if not valid_indices: continue
            x_plot = [current_x_vals[i] for i in valid_indices]
            y_plot = subdf["total_mean"].iloc[valid_indices].values
            y_err_plot = subdf["total_std"].iloc[valid_indices].fillna(0).values
            ax.errorbar(
                x=x_plot, y=y_plot, yerr=y_err_plot,
                fmt="none", capsize=4, color='black', alpha=0.6, linewidth=1
            )
        ax.set_xlabel("Prompt Style")
        ax.set_ylabel("Total Score")
        ax.set_title(f"{scale_name} Total Scores")
        ax.tick_params(axis='x', rotation=15)
        ax.legend(title="Model")
        plt.tight_layout()
        plt.show()

    rwa_plot_data = combined_df[combined_df["scale"] == "RWA"]
    if not rwa_plot_data.empty:
        create_single_plot(rwa_plot_data, "RWA")
    else:
        print("No RWA data found to plot.")

    lwa_plot_data = combined_df[combined_df["scale"] == "LWA"]
    if not lwa_plot_data.empty:
        create_single_plot(lwa_plot_data, "LWA")
    else:
        print("No LWA data found to plot.") 