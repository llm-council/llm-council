import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lm_council.constants import FAMILY_COLORS


def sorted_dict_of_dict(data):
    """Returns a sorted dictionary of dictionaries, ensuring that councils are at the end."""
    # Sort the outer dictionary keys
    sorted_outer_keys = sorted(data.keys())

    # First find all possible inner keys.
    # In the completer vs. completer scenario, the inner keys are different for each outer key
    # because each completer does not face off against itself.
    all_inner_keys = set()
    for outer_key in sorted_outer_keys:
        all_inner_keys.update(data[outer_key].keys())

    sorted_inner_keys = sorted(list(all_inner_keys))

    # Ensures that councils are at the end.
    if "council" in sorted_outer_keys:
        sorted_outer_keys.remove("council")
        sorted_outer_keys.append("council")
    if "council" in sorted_inner_keys:
        sorted_inner_keys.remove("council")
        sorted_inner_keys.append("council")
    if "council/majority-vote" in sorted_outer_keys:
        sorted_outer_keys.remove("council/majority-vote")
        sorted_outer_keys.append("council/majority-vote")
    if "council/mean-pooling" in sorted_outer_keys:
        sorted_outer_keys.remove("council/mean-pooling")
        sorted_outer_keys.append("council/mean-pooling")
    if "council/no-aggregation" in sorted_outer_keys:
        sorted_outer_keys.remove("council/no-aggregation")
        sorted_outer_keys.append("council/no-aggregation")
    if "council/majority-vote" in sorted_inner_keys:
        sorted_inner_keys.remove("council/majority-vote")
        sorted_inner_keys.append("council/majority-vote")
    if "council/mean-pooling" in sorted_inner_keys:
        sorted_inner_keys.remove("council/mean-pooling")
        sorted_inner_keys.append("council/mean-pooling")
    if "council/no-aggregation" in sorted_inner_keys:
        sorted_inner_keys.remove("council/no-aggregation")
        sorted_inner_keys.append("council/no-aggregation")

    final_structure = {}
    for outer_key in sorted_outer_keys:
        # Sort the inner dictionary by the sorted outer keys for consistent ordering
        final_structure[outer_key] = {
            inner_key: data[outer_key].get(inner_key) for inner_key in sorted_inner_keys
        }

    return final_structure


def plot_heatmap(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    vmin: float = None,
    vmax: float = None,
    center: float = None,
    cmap: str = "coolwarm",
    outfile: str = None,
    figsize: tuple[int, int] = (20, 16),
    fmt: str = ".2f",
    font_size: int = 18,
    title: str = None,
):
    """Plots a heatmap of the data. Saves the plot to a file if outfile is provided."""
    # Change font size globally.
    plt.rcParams.update({"font.size": font_size})

    # Dynamically determine figure size if figsize is None
    if figsize is None:
        nrows, ncols = df.shape

        # Linear interpolation between (2,2)->(4,4) and (20,20)->(20,20)
        def interp(val, x0, x1, y0, y1):
            if val <= x0:
                return y0
            if val >= x1:
                return y1
            return y0 + (y1 - y0) * (val - x0) / (x1 - x0)

        width = interp(ncols, 2, 20, 4, 20)
        height = interp(nrows, 2, 20, 4, 16)
        figsize = (width, height)

    # Plotting the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        df,
        annot=df,
        cmap=cmap,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax,
        center=center,
    )
    if title:
        plt.title(title)
    else:
        plt.title("")
    plt.setp(plt.xticks()[1], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile)
    plt.show()
    plt.close()


def plot_arena_hard_elo_stats(stats, title, outfile, show=False):
    # Mapping and error calculations
    stats["family"] = stats["model"].apply(lambda x: x.split("/")[0])
    stats["lower_error"] = stats["score"] - stats["lower"]
    stats["upper_error"] = stats["upper"] - stats["score"]
    errors = stats[["lower_error", "upper_error"]].T.values

    # Create the plot with larger fonts
    # Dynamically set figure height based on number of models (entries)
    n_models = len(stats)
    height = max(4, min(0.6 * n_models + 2, 16))  # min 4, max 16, scale with n_models
    plt.figure(figsize=(10, height))

    # Increase font size globally
    plt.rcParams.update({"font.size": 10})

    ax = sns.barplot(
        data=stats,
        y="model",
        x="score",
        hue="family",
        palette=FAMILY_COLORS,
    )

    # Add error bars only if lower_error != upper_error
    for i, (score, error) in enumerate(zip(stats["score"], errors.T)):
        lower_err, upper_err = error[0], error[1]
        if lower_err != upper_err:
            try:
                ax.errorbar(
                    score,
                    i,
                    xerr=[[lower_err], [upper_err]],
                    fmt="none",
                    c="black",
                    capsize=5,
                )
            except Exception as e:
                print(
                    f"Error bars aren't valid: {error[:1], error[1:]}. This can happen due to bad bootstrap sampling, particularly in low data scenarios. Skipping."
                )
        plt.text(score + upper_err + 1, i, f"{score}", va="center", color="black")

    plt.title(title)
    plt.xlabel("Win Rate")
    plt.ylabel("")
    plt.grid(axis="x", linestyle="--")
    plt.legend(title="Family", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.xlim(right=stats["score"].max() + 20)
    plt.tight_layout()

    # Save to file if specified
    if outfile:
        plt.savefig(outfile)
        print(f"Arena leaderboard saved to {outfile}.")
    if show:
        plt.show()
    plt.close()


def get_grouped_scores(judging_df, eval_config):
    """Groups the judging DataFrame by model and computes aggregated scores for each criterion."""

    criteria_names = [c.name for c in eval_config.config.rubric] + ["Overall"]
    # Build aggregation functions for all criteria
    agg_funcs = {name: ["mean", list, "std"] for name in criteria_names}
    grouped = judging_df.groupby("model_being_judged").agg(agg_funcs)
    # Flatten MultiIndex columns
    grouped.columns = [
        f"{col[0]}{'' if col[1]=='mean' else '__'+col[1]}" for col in grouped.columns
    ]

    # # Compute Overall and Overall__std
    # # def overall(row):
    # #     return sum(row[name] for name in criteria_names) / len(criteria_names)

    # # def overall_raw(row):
    # #     # Zip the raw lists and average per judge
    # #     return [
    # #         sum(vals) / len(vals)
    # #         for vals in zip(*(row[f"{name}__list"] for name in criteria_names))
    # #     ]

    # # grouped["Overall"] = grouped.apply(overall, axis=1)
    # # grouped["Overall__raw"] = grouped.apply(overall_raw, axis=1)
    # # grouped["Overall__std"] = grouped["Overall__raw"].apply(
    #     lambda x: pd.Series(x).std()
    # )

    # Add family column
    grouped = grouped.reset_index()
    grouped["family"] = grouped["model_being_judged"].apply(lambda x: x.split("/")[0])
    return grouped


def plot_direct_assessment_charts(judging_df, eval_config, outfile=None):
    grouped = get_grouped_scores(judging_df, eval_config)

    criteria_names = [c.name for c in eval_config.config.rubric]
    # Put "Overall" first
    plot_names = ["Overall"] + criteria_names
    n_plots = len(plot_names)
    nrows, ncols = 3, 3
    total_subplots = nrows * ncols

    # Create the figure and axes with shared x and y axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Determine y-tick labels order (sorted by overall)
    yticklabels = grouped.sort_values(by="Overall", ascending=False)[
        "model_being_judged"
    ]

    for idx, name in enumerate(plot_names):
        ax = axes[idx]
        data = grouped.set_index("model_being_judged").loc[yticklabels].reset_index()
        sns.barplot(
            data=data,
            x=name,
            y="model_being_judged",
            hue="family",
            palette=FAMILY_COLORS,
            orient="h",
            ax=ax,
        )
        # Add bar annotations
        for p in ax.patches:
            width = p.get_width()
            if width:
                ax.annotate(
                    f"{width:.2f}",
                    (width, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=9,
                )
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[1] + 0.03 * (xlim[1] - xlim[0]))
        ax.set_title(f"{name if name != 'Overall' else 'Overall Scores'}")
        ax.set_xlabel("Mean Score" if name != "Overall" else "Mean Overall Score")
        ax.set_ylabel("")

    # Hide unused subplots
    for i in range(n_plots, total_subplots):
        fig.delaxes(axes[i])

    # Add a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Family",
        loc="upper center",
        ncol=len(labels),
    )

    # Remove individual legends from each subplot.
    for ax in axes:
        ax.legend_.remove() if ax.get_legend() else None

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend at the top

    # Save to file if specified
    if outfile:
        plt.savefig(outfile)
        print(f"Direct assessment leaderboard saved to {outfile}.")
    else:
        plt.show()

    return grouped
