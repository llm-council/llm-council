import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


FAMILY_COLORS = {
    "openai": "mediumseagreen",
    "anthropic": "burlywood",
    "mistral": "darkorange",
    "google": "skyblue",
    "meta-llama": "magenta",
    "deepseek": "royalblue",
    "cohere": "darkslategray",
    "qwen": "slateblue",
    "council": "gold",
    "amazon": "orange",
    "x-ai": "black",
    "01-ai": "teal",
    "recursal": "darkslateblue",
}


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
    plt.rcParams.update({"font.size": 12})

    ax = sns.barplot(
        data=stats,
        y="model",
        x="score",
        hue="family",
        palette=FAMILY_COLORS,
    )

    # Add error bars
    for i, (score, error) in enumerate(zip(stats["score"], errors.T)):
        try:
            ax.errorbar(
                score, i, xerr=[error[:1], error[1:]], fmt="none", c="black", capsize=5
            )
        except Exception as e:
            print(
                f"Error bars aren't valid: {error[:1], error[1:]}. This can happen due to bad bootstrap sampling, particularly in low data scenarios. Skipping."
            )
        plt.text(score + error[1:] + 1, i, f"{score}", va="center", color="black")

    plt.title(title)
    plt.xlabel("Win Rate")
    plt.ylabel("")
    plt.grid(axis="x", linestyle="--")
    plt.legend(title="Family", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=16)
    plt.xlim(right=stats["score"].max() + 20)
    plt.tight_layout()

    # Save to file if specified
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()
    plt.close()
