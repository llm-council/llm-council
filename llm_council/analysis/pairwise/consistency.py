import pandas as pd
from collections import defaultdict
from llm_council.constants import (
    MAJOR_A_WIN,
    MINOR_A_WIN,
    MINOR_B_WIN,
    MAJOR_B_WIN,
    TIE,
)


def consistency_with_sidewise_tolerance_fn(pairwise_choice, position_reversed_choice):
    """Returns whether the rating couplet is consistent.

    The couplet is consistent as long as the sides of the two choices (A/B) are consistent.

    Fine-grained differences like A>>B and B>A are still considered consistent.
    Ties are only considered consistent with ties.
    """
    return (
        (pairwise_choice == MINOR_A_WIN and position_reversed_choice == MINOR_B_WIN)
        or (pairwise_choice == MAJOR_A_WIN and position_reversed_choice == MAJOR_B_WIN)
        or (pairwise_choice == MAJOR_A_WIN and position_reversed_choice == MINOR_B_WIN)
        or (pairwise_choice == MINOR_A_WIN and position_reversed_choice == MAJOR_B_WIN)
        or (pairwise_choice == MINOR_B_WIN and position_reversed_choice == MINOR_A_WIN)
        or (pairwise_choice == MINOR_B_WIN and position_reversed_choice == MAJOR_A_WIN)
        or (pairwise_choice == MAJOR_B_WIN and position_reversed_choice == MINOR_A_WIN)
        or (pairwise_choice == MAJOR_B_WIN and position_reversed_choice == MAJOR_A_WIN)
        or (pairwise_choice == TIE and position_reversed_choice == TIE)
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
    example_id_column_name="emobench_id",
    consistency_fn_name="sidewise",
) -> pd.DataFrame:
    """Returns a DataFrame of consistent votes."""

    # Get the unique judges.
    judges = list(df["llm_judge"].unique())

    # Create a map of judge -> (id, first, second) -> row.
    judge_to_votes_map = defaultdict(dict)
    for i, row in df.iterrows():
        judge_to_votes_map[row["llm_judge"]][
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
