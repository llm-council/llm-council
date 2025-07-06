from collections import defaultdict

import pandas as pd

from lm_council.analysis.pairwise.pairwise_utils import get_council_choice
from lm_council.analysis.pairwise.separability import (
    compute_mle_elo,
    filter_ratings_by_allowlist,
    get_win_rate,
)
from lm_council.analysis.visualization import sorted_dict_of_dict


def get_affinity_df(
    judging_df, reference_llm_respondent, example_id_column="emobench_id"
) -> dict[str, pd.DataFrame]:
    """Returns a dictionary of dataframes.

    3 dataframes are returned:
    - judge_preferences: judge -> participant -> affinity
    - judge_preferences_council_normalized: judge -> participant -> (affinity - council_affinity)
    - self_enhancement_bias: judge -> bias
    """
    per_judge_preferences = {}
    council_members = list(judging_df["judge_model"].unique())
    for council_member in council_members:
        filtered_df = filter_ratings_by_allowlist(judging_df, [council_member])

        # Single instance, no variation.
        bootstrap_online_elo = compute_mle_elo(filtered_df, reference_llm_respondent)

        # Get win rates.
        win_rates = get_win_rate(bootstrap_online_elo, reference_llm_respondent)

        per_judge_preferences[council_member] = win_rates

    # Add the council.
    council_choice = get_council_choice(judging_df, "majority", example_id_column)
    bootstrap_online_elo = compute_mle_elo(council_choice, reference_llm_respondent)
    win_rates = get_win_rate(bootstrap_online_elo, reference_llm_respondent)
    per_judge_preferences["council/majority-vote"] = win_rates

    council_choice = get_council_choice(judging_df, "mean_pooling", example_id_column)
    bootstrap_online_elo = compute_mle_elo(council_choice, reference_llm_respondent)
    win_rates = get_win_rate(bootstrap_online_elo, reference_llm_respondent)
    per_judge_preferences["council/mean-pooling"] = win_rates

    council_choice = judging_df
    bootstrap_online_elo = compute_mle_elo(council_choice, reference_llm_respondent)
    win_rates = get_win_rate(bootstrap_online_elo, reference_llm_respondent)
    per_judge_preferences["council/no-aggregation"] = win_rates

    per_judge_preferences_df = pd.DataFrame(sorted_dict_of_dict(per_judge_preferences))
    per_judge_preferences_df = per_judge_preferences_df / 100

    # Subtract out the council's win rate.
    per_judge_preferences_council_normalized = defaultdict(dict)
    for judge, participant_affinity in per_judge_preferences.items():
        if judge == "council/no-aggregation":
            continue
        for participant, affinity in participant_affinity.items():
            per_judge_preferences_council_normalized[judge][participant] = (
                affinity - per_judge_preferences["council/no-aggregation"][participant]
            )
    per_judge_preferences_council_normalized_df = pd.DataFrame(
        per_judge_preferences_council_normalized
    )
    per_judge_preferences_council_normalized_df = (
        per_judge_preferences_council_normalized_df / 100
    )

    # Self-enhancement bias.
    self_enhancement_bias = {}
    for judge in council_members:
        if judge in per_judge_preferences_council_normalized[judge]:
            self_enhancement_bias[judge] = per_judge_preferences_council_normalized[
                judge
            ][judge]
    self_enhancement_bias_df = pd.DataFrame(self_enhancement_bias, index=["bias"])
    self_enhancement_bias_df = self_enhancement_bias_df / 100

    return {
        "judge_preferences": per_judge_preferences_df,
        "judge_preferences_council_normalized": per_judge_preferences_council_normalized_df,
        "self_enhancement_bias": self_enhancement_bias_df,
    }
