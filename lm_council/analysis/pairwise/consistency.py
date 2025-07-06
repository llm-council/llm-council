from collections import defaultdict

import numpy as np
import pandas as pd

from lm_council.analysis.pairwise.agreement import get_side
from lm_council.analysis.pairwise.pairwise_utils import get_council_choice
from lm_council.constants import MAJOR_A_WIN, MAJOR_B_WIN, MINOR_A_WIN, MINOR_B_WIN, TIE


def consistency_with_sidewise_tolerance_fn(pairwise_choice, position_reversed_choice):
    """Returns whether the rating couplet is consistent.

    The couplet is consistent as long as the sides of the two choices (A/B) are consistent.

    Fine-grained differences like A>>B and B>A are still considered consistent.
    Ties are only considered consistent with ties.
    """
    pairwise_choice_side = get_side(pairwise_choice)
    position_reversed_choice_side = get_side(position_reversed_choice)
    return (
        (pairwise_choice_side == "A" and position_reversed_choice_side == "B")
        or (pairwise_choice_side == "B" and position_reversed_choice_side == "A")
        or (pairwise_choice_side == TIE and position_reversed_choice_side == TIE)
    )


def consistency_strict_fn(pairwise_choice, position_reversed_choice):
    """Returns whether the rating couplet is consistent.

    Fine-grained differences like A>>B and B>A are considered inconsistent.
    Ties are only considered consistent with ties.
    """
    return (
        (pairwise_choice == MINOR_A_WIN and position_reversed_choice == MINOR_B_WIN)
        or (pairwise_choice == MAJOR_A_WIN and position_reversed_choice == MAJOR_B_WIN)
        or (pairwise_choice == MINOR_B_WIN and position_reversed_choice == MINOR_A_WIN)
        or (pairwise_choice == MAJOR_B_WIN and position_reversed_choice == MAJOR_A_WIN)
        or (pairwise_choice == TIE and position_reversed_choice == TIE)
    )


CONSISTENCY_FN_REGISTRY = {
    "sidewise": consistency_with_sidewise_tolerance_fn,
    "strict": consistency_strict_fn,
}


def get_consistent_votes(
    df,
    example_id_column_name,
    consistency_fn_name="sidewise",
    first_completion_by_column_name="first_completion_by",
    second_completion_by_column_name="second_completion_by",
    pairwise_choice_column_name="pairwise_choice",
    llm_judge_column_name="judge_model",
) -> pd.DataFrame:
    """Returns a DataFrame of consistent votes."""

    # Get the unique judges.
    judges = list(df["judge_model"].unique())

    if example_id_column_name is None:
        # Use the index.
        example_id_column_name = df.index.name

    # Create a map of judge -> (id, first, second) -> row.
    judge_to_votes_map = defaultdict(dict)
    for i, row in df.iterrows():
        judge_to_votes_map[row["judge_model"]][
            (
                row[example_id_column_name],
                row["first_completion_by"],
                row["second_completion_by"],
            )
        ] = row

    # Get the consistency function.
    consistency_fn = CONSISTENCY_FN_REGISTRY[consistency_fn_name]

    consistent_votes = []
    for judge, votes in judge_to_votes_map.items():
        vote_keys = votes.keys()

        for vote_key in vote_keys:
            (id, first_completion_by, second_completion_by) = vote_key

            # Look up the pairwise choice and the reverse choice.
            pairwise_choice = votes[vote_key]["pairwise_choice"]
            if (id, second_completion_by, first_completion_by) not in vote_keys:
                # Votes without a reverse choice are consistent by default (there is no counter-vote).
                consistent_votes.append(votes[vote_key].to_dict())
                continue

            position_reversed_choice = votes[
                (id, second_completion_by, first_completion_by)
            ]["pairwise_choice"]

            if consistency_fn(pairwise_choice, position_reversed_choice):
                consistent_votes.append(votes[vote_key])

    return pd.DataFrame(consistent_votes)


def get_number_of_mode(pd_agg):
    """For a given pandas dataframe aggregation (list), returns the number of occurrences of the mode."""
    votes = list(pd_agg)
    mode = pd_agg.mode()[0]
    num_occurrences = 0
    for vote in votes:
        if vote == mode:
            num_occurrences += 1
    return num_occurrences


def get_judge_invariability_df(
    full_judging_df, example_id_column="emobench_id"
) -> pd.DataFrame:
    consolidated_reps = (
        full_judging_df.groupby(
            [
                example_id_column,
                "judge_model",
                "first_completion_by",
                "second_completion_by",
            ]
        )
        .agg(
            # The most common pairwise choice over reps.
            pairwise_choice_mode=pd.NamedAgg(
                column="pairwise_choice", aggfunc=lambda x: x.mode()[0]
            ),
            # The number of reps.
            num_reps=pd.NamedAgg(column="pairwise_choice", aggfunc=lambda x: len(x)),
            # All votes.
            all_votes=pd.NamedAgg(column="pairwise_choice", aggfunc=list),
            # The number of occurrences of the mode.
            number_of_mode=pd.NamedAgg(
                column="pairwise_choice", aggfunc=get_number_of_mode
            ),
            # The number of unique responses.
            number_of_unique_responses=pd.NamedAgg(
                "pairwise_choice", aggfunc=lambda x: len(set(x))
            ),
        )
        .reset_index()
    )

    consolidated_reps["invariability"] = (
        consolidated_reps["number_of_mode"] / consolidated_reps["num_reps"]
    )
    return consolidated_reps


def get_judge_invariability_summary_df(conslidated_reps_df) -> pd.DataFrame:
    return conslidated_reps_df.groupby("judge_model").agg(
        mean_invariability=pd.NamedAgg(column="invariability", aggfunc="mean"),
        mean_num_unique_responses=pd.NamedAgg(
            column="number_of_unique_responses", aggfunc="mean"
        ),
    )


def get_judge_profile_stats(judge_choice, example_id_column="emobench_id"):
    # Collapse pure repetitions via mode.
    repetition_collapsed_judging_df = (
        judge_choice.groupby(
            [example_id_column, "first_completion_by", "second_completion_by"]
        )
        .agg(
            pairwise_choice=pd.NamedAgg(
                column="pairwise_choice", aggfunc=lambda x: x.mode()[0]
            ),
        )
        .reset_index()
    )
    # Aggregate pure repetitions by list.
    repetition_aggregated_judging_df = (
        judge_choice.groupby(
            [example_id_column, "first_completion_by", "second_completion_by"]
        )
        .agg(
            pairwise_choice=pd.NamedAgg(column="pairwise_choice", aggfunc=list),
        )
        .reset_index()
    )

    # Build a map of (id, a, b) -> choice.
    votes_map = defaultdict(dict)
    for i, row in repetition_collapsed_judging_df.iterrows():
        votes_map[
            (
                row[example_id_column],
                row["first_completion_by"],
                row["second_completion_by"],
            )
        ] = row["pairwise_choice"]

    # Build a map of (id, a, b) -> list of choices.
    repped_votes_map = defaultdict(list)
    for i, row in repetition_aggregated_judging_df.iterrows():
        repped_votes_map[
            (
                row[example_id_column],
                row["first_completion_by"],
                row["second_completion_by"],
            )
        ] = row["pairwise_choice"]

    consistent_votes = 0
    inconsistent_votes = 0
    bias_towards_a = 0
    bias_towards_b = 0
    consistent_a_votes = 0
    consistent_b_votes = 0
    consistent_tie_votes = 0
    aggregated_consistencies = []

    for (id, first_completion_by, second_completion_by), rating in votes_map.items():
        reverse_rating_id = (id, second_completion_by, first_completion_by)
        if reverse_rating_id not in votes_map:
            # Skipping votes that don't have a reverse rating, which may happen during no-tie analysis.
            continue

        reverse_rating = votes_map[(id, second_completion_by, first_completion_by)]

        side_of_rating = get_side(rating)
        if consistency_with_sidewise_tolerance_fn(rating, reverse_rating):
            consistent_votes += 1
            if side_of_rating == "A":
                consistent_a_votes += 1
            elif side_of_rating == "B":
                consistent_b_votes += 1
            else:
                consistent_tie_votes += 1
        else:
            inconsistent_votes += 1
            if side_of_rating == "A":
                bias_towards_a += 1
            elif side_of_rating == "B":
                bias_towards_b += 1

        # Calculate list-aggregated consistency.
        def is_consistent_numeric(rating_tuple):
            rating, reverse_rating = rating_tuple
            if consistency_with_sidewise_tolerance_fn(rating, reverse_rating):
                return 1
            return 0

        repped_ratings = repped_votes_map[
            (id, first_completion_by, second_completion_by)
        ]
        repped_reverse_ratings = repped_votes_map[
            (id, second_completion_by, first_completion_by)
        ]
        couplets = [(x, y) for x in repped_ratings for y in repped_reverse_ratings]

        # Count the number of consistent couplets for this (id, a, b).
        consistent_couplet_counter = 0
        for couplet in couplets:
            consistent_couplet_counter += is_consistent_numeric(couplet)

        # Register the percentage of consistent couplets.
        aggregated_consistencies.append(consistent_couplet_counter / len(couplets))

    rating_counts = judge_choice["pairwise_choice"].value_counts()
    major_a_votes = rating_counts.get(MAJOR_A_WIN, 0)
    minor_a_votes = rating_counts.get(MINOR_A_WIN, 0)
    major_b_votes = rating_counts.get(MAJOR_B_WIN, 0)
    minor_b_votes = rating_counts.get(MINOR_B_WIN, 0)
    tie_votes = rating_counts.get(TIE, 0)
    num_votes = len(votes_map.keys())

    return {
        "inconsistency": inconsistent_votes / num_votes,
        "consistency": consistent_votes / num_votes,
        "repped_consistency": np.mean(aggregated_consistencies),
        "consistent_a_votes": consistent_a_votes / num_votes,
        "consistent_b_votes": consistent_b_votes / num_votes,
        "consistent_tie_votes": consistent_tie_votes / num_votes,
        "bias_towards_a": bias_towards_a / num_votes,
        "bias_towards_b": bias_towards_b / num_votes,
        "a_votes": (major_a_votes + minor_a_votes) / num_votes,
        "b_votes": (major_b_votes + minor_b_votes) / num_votes,
        "major_votes": (major_a_votes + major_b_votes) / num_votes,
        "minor_votes": (minor_a_votes + minor_b_votes) / num_votes,
        "major_a_votes": major_a_votes / num_votes,
        "minor_a_votes": minor_a_votes / num_votes,
        "major_b_votes": major_b_votes / num_votes,
        "minor_b_votes": minor_b_votes / num_votes,
        "tie_votes": tie_votes / num_votes,
    }


def get_judge_profiles_df(judging_df):
    # Individual judges.
    judge_to_rating_profile_stats = {}
    for judge in judging_df["judge_model"].unique():
        judge_choice = judging_df[judging_df["judge_model"] == judge]
        judge_to_rating_profile_stats[judge] = get_judge_profile_stats(judge_choice)

    # Add the council.
    council_choice = get_council_choice(judging_df, "majority")
    judge_to_rating_profile_stats["council/majority-vote"] = get_judge_profile_stats(
        council_choice
    )
    council_choice = get_council_choice(judging_df, "mean_pooling")
    judge_to_rating_profile_stats["council/mean-pooling"] = get_judge_profile_stats(
        council_choice
    )

    return pd.DataFrame(judge_to_rating_profile_stats).T
