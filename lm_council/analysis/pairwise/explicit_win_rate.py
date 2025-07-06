from collections import defaultdict

import pandas as pd

from lm_council.analysis.visualization import sorted_dict_of_dict
from lm_council.constants import MAJOR_A_WIN, MAJOR_B_WIN, MINOR_A_WIN, MINOR_B_WIN, TIE


def get_llm_respondent_vs_respondent_stats(df, major_win_multiplier=3):
    """Returns a dictionary of respondent vs. respondent stats.

    Strong wins are scaled by the major_win_multiplier.
    """
    # llm_completer -> llm_completer -> dict of stats.
    llm_respondent_vs_respondent_stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    respondents = list(df["first_completion_by"].unique())

    # (llm1, llm2, choice) -> count
    completer_vs_completer_ratings = (
        df.groupby(["first_completion_by", "second_completion_by"])["pairwise_choice"]
        .value_counts()
        .to_dict()
    )

    for (
        respondent_1,
        respondent_2,
        rating,
    ), count in completer_vs_completer_ratings.items():
        if rating == MINOR_A_WIN:
            llm_respondent_vs_respondent_stats[respondent_1][respondent_2][
                "num_wins"
            ] += count
            llm_respondent_vs_respondent_stats[respondent_2][respondent_1][
                "num_losses"
            ] += count
        if rating == MAJOR_A_WIN:
            llm_respondent_vs_respondent_stats[respondent_1][respondent_2][
                "num_wins"
            ] += (count * major_win_multiplier)
            llm_respondent_vs_respondent_stats[respondent_2][respondent_1][
                "num_losses"
            ] += (count * major_win_multiplier)
        if rating == MINOR_B_WIN:
            llm_respondent_vs_respondent_stats[respondent_2][respondent_1][
                "num_wins"
            ] += count
            llm_respondent_vs_respondent_stats[respondent_1][respondent_2][
                "num_losses"
            ] += count
        if rating == MAJOR_B_WIN:
            llm_respondent_vs_respondent_stats[respondent_2][respondent_1][
                "num_wins"
            ] += (count * major_win_multiplier)
            llm_respondent_vs_respondent_stats[respondent_1][respondent_2][
                "num_losses"
            ] += (count * major_win_multiplier)
        if rating == TIE:
            llm_respondent_vs_respondent_stats[respondent_1][respondent_2][
                "num_ties"
            ] += count
            llm_respondent_vs_respondent_stats[respondent_2][respondent_1][
                "num_ties"
            ] += count
    return llm_respondent_vs_respondent_stats


def get_explicit_win_rates(df):
    """Returns a dataframe of the explicit win rates between completers."""
    llm_respondent_vs_respondent_stats = get_llm_respondent_vs_respondent_stats(df)

    # Straightforward count.
    llm_completer_to_completer_win_rate = defaultdict(lambda: defaultdict(float))
    for (
        first_llm_completer,
        second_llm_completer_to_stats,
    ) in llm_respondent_vs_respondent_stats.items():
        for second_llm_completer, stats in second_llm_completer_to_stats.items():
            # Skip head to head.
            if first_llm_completer == second_llm_completer:
                continue
            # Skip if no games played.
            if stats["num_wins"] + stats["num_ties"] + stats["num_losses"] == 0:
                continue
            # We swap 2nd and 1st completers here so that the X-axis of the plot can be the "vs." option.
            llm_completer_to_completer_win_rate[second_llm_completer][
                first_llm_completer
            ] = stats["num_wins"] / (
                stats["num_wins"] + stats["num_ties"] + stats["num_losses"]
            )
    return pd.DataFrame(sorted_dict_of_dict(llm_completer_to_completer_win_rate))
