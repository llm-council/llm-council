import os
from collections import defaultdict

import choix
import numpy as np
import pandas as pd

from lm_council.analysis.visualization import sorted_dict_of_dict
from lm_council.constants import MAJOR_A_WIN, MAJOR_B_WIN, MINOR_A_WIN, MINOR_B_WIN, TIE


def get_choix_data(df, major_win_multiplier=3):
    """Returns a list of battle outcomes for the choix library to compute Bradley-Terry outcomes."""
    subjects = np.unique(df[["first_completion_by", "second_completion_by"]])
    n_items = len(subjects)
    council_member_map = {subjects[i]: i for i in range(n_items)}

    data = []
    for i, row in df.iterrows():
        if row["pairwise_choice"] == MAJOR_A_WIN:
            for _ in range(major_win_multiplier):
                data.append(
                    (
                        council_member_map[row["first_completion_by"]],
                        council_member_map[row["second_completion_by"]],
                    )
                )
        if row["pairwise_choice"] == MINOR_A_WIN:
            data.append(
                (
                    council_member_map[row["first_completion_by"]],
                    council_member_map[row["second_completion_by"]],
                )
            )
        if row["pairwise_choice"] == MAJOR_B_WIN:
            for _ in range(major_win_multiplier):
                data.append(
                    (
                        council_member_map[row["second_completion_by"]],
                        council_member_map[row["first_completion_by"]],
                    )
                )
        if row["pairwise_choice"] == MINOR_B_WIN:
            data.append(
                (
                    council_member_map[row["second_completion_by"]],
                    council_member_map[row["first_completion_by"]],
                )
            )
    return data


def bradley_terry_analysis(df):
    """Returns the expected win rate map estimated using Bradley-Terry."""
    subjects = np.unique(df[["first_completion_by", "second_completion_by"]])
    n_items = len(subjects)
    council_member_map = {subjects[i]: i for i in range(n_items)}

    data = get_choix_data(df)
    params = choix.ilsr_pairwise(n_items, data)

    # Prep the expected win rate map for the heatmap.
    expected_win_rate_map = defaultdict(dict)
    for council_member_1 in subjects:
        for council_member_2 in subjects:
            prob_1_wins, prob_2_wins = choix.probabilities(
                [
                    council_member_map[council_member_1],
                    council_member_map[council_member_2],
                ],
                params,
            )
            # We reverse the order to ensure the heatmap rendering can be read from left to right.
            expected_win_rate_map[council_member_2][council_member_1] = prob_1_wins
            expected_win_rate_map[council_member_1][council_member_2] = prob_2_wins

    return pd.DataFrame(sorted_dict_of_dict(expected_win_rate_map))
