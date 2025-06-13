import pandas as pd
import seaborn as sns
from llm_council.members.membership import FULLY_QUALIFIED_NAME_TO_LLM_MAP
import matplotlib.pyplot as plt


def get_plot_friendly_name(fully_qualified_name):
    if fully_qualified_name in FULLY_QUALIFIED_NAME_TO_LLM_MAP:
        return FULLY_QUALIFIED_NAME_TO_LLM_MAP[fully_qualified_name]
    else:
        return fully_qualified_name


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
    if "council (by majority vote)" in sorted_outer_keys:
        sorted_outer_keys.remove("council (by majority vote)")
        sorted_outer_keys.append("council (by majority vote)")
    if "council (by mean pooling)" in sorted_outer_keys:
        sorted_outer_keys.remove("council (by mean pooling)")
        sorted_outer_keys.append("council (by mean pooling)")
    if "council (no aggregation)" in sorted_outer_keys:
        sorted_outer_keys.remove("council (no aggregation)")
        sorted_outer_keys.append("council (no aggregation)")
    if "council (by majority vote)" in sorted_inner_keys:
        sorted_inner_keys.remove("council (by majority vote)")
        sorted_inner_keys.append("council (by majority vote)")
    if "council (by mean pooling)" in sorted_inner_keys:
        sorted_inner_keys.remove("council (by mean pooling)")
        sorted_inner_keys.append("council (by mean pooling)")
    if "council (no aggregation)" in sorted_inner_keys:
        sorted_inner_keys.remove("council (no aggregation)")
        sorted_inner_keys.append("council (no aggregation)")

    final_structure = {}
    for outer_key in sorted_outer_keys:
        # Sort the inner dictionary by the sorted outer keys for consistent ordering
        final_structure[get_plot_friendly_name(outer_key)] = {
            get_plot_friendly_name(inner_key): data[outer_key].get(inner_key)
            for inner_key in sorted_inner_keys
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
