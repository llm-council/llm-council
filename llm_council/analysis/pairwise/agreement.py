import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from collections import defaultdict
from llm_council.constants import (
    MAJOR_A_WIN,
    MINOR_A_WIN,
    MINOR_B_WIN,
    MAJOR_B_WIN,
    TIE,
)
from llm_council.analysis.visualization import sorted_dict_of_dict
from llm_council.analysis.pairwise.pairwise_utils import get_council_choice


AGREEMENT_METHODS = ["cohen_kappa", "exact", "sidewise", "sidewise_cohen_kappa"]


def get_side(rating):
    if rating == "A>B" or rating == "A>>B":
        return "A"
    if rating == "B>A" or rating == "B>>A":
        return "B"
    return "Tie"


def get_judge_to_judge_agreement(judge1_votes, judge2_votes, method):
    agreements = []
    judge1_vote_keys = set(judge1_votes.keys())
    judge2_vote_keys = set(judge2_votes.keys())

    # If this is after removing inconsistent votes, it's possible that many votes are
    # non-overlapping.
    both_judge_vote_keys = judge1_vote_keys & judge2_vote_keys

    if not both_judge_vote_keys:
        return None

    # Extract the responses from both judges
    responses_judge1 = [judge1_votes[qid] for qid in both_judge_vote_keys]
    responses_judge2 = [judge2_votes[qid] for qid in both_judge_vote_keys]
    if method == "cohen_kappa":
        return cohen_kappa_score(
            responses_judge1,
            responses_judge2,
            labels=[MINOR_A_WIN, MINOR_B_WIN, MAJOR_A_WIN, MAJOR_B_WIN],
        )
    elif method == "exact":
        num_agreeements = 0
        for response_judge1, response_judge2 in zip(responses_judge1, responses_judge2):
            if response_judge1 == response_judge2:
                num_agreeements += 1
        return num_agreeements / len(responses_judge1)
    elif method == "sidewise":
        num_agreeements = 0
        for response_judge1, response_judge2 in zip(responses_judge1, responses_judge2):
            if (
                get_side(response_judge1) == get_side(response_judge2)
                or get_side(response_judge1) == "Tie"
                or get_side(response_judge2) == "Tie"
            ):
                num_agreeements += 1
        return num_agreeements / len(responses_judge1)
    elif method == "sidewise_cohen_kappa":
        return cohen_kappa_score(
            [get_side(response) for response in responses_judge1],
            [get_side(response) for response in responses_judge2],
            labels=["A", "B"],
        )
    else:
        raise ValueError(f"Unknown judge_to_judge_agreement method: {method}")


def get_judge_agreement_df(df, agreement_method):
    judges = list(df["llm_judge"].unique())

    judge_to_votes_map = defaultdict(dict)
    for i, row in df.iterrows():
        judge_to_votes_map[row["llm_judge"]][
            (
                row["emobench_id"],
                row["first_completion_by"],
                row["second_completion_by"],
            )
        ] = row["pairwise_choice"]

    # Add the council.
    council_choice = get_council_choice(df, "majority")
    for i, row in council_choice.iterrows():
        judge_to_votes_map["council (by majority vote)"][
            (
                row["emobench_id"],
                row["first_completion_by"],
                row["second_completion_by"],
            )
        ] = row["pairwise_choice"]
    council_choice = get_council_choice(df, "mean_pooling")
    for i, row in council_choice.iterrows():
        judge_to_votes_map["council (by mean pooling)"][
            (
                row["emobench_id"],
                row["first_completion_by"],
                row["second_completion_by"],
            )
        ] = row["pairwise_choice"]

    # judge -> judge -> agreement (float)
    judge_to_judge_agreement = defaultdict(dict)
    for judge1 in judges + ["council (by majority vote)", "council (by mean pooling)"]:
        judge1_votes = judge_to_votes_map[judge1]

        for judge2 in judges + [
            "council (by majority vote)",
            "council (by mean pooling)",
        ]:
            if judge1 == judge2:
                continue
            judge2_votes = judge_to_votes_map[judge2]

            agreement = get_judge_to_judge_agreement(
                judge1_votes, judge2_votes, agreement_method
            )
            if agreement is not None:
                judge_to_judge_agreement[judge1][judge2] = agreement

    # Fix the LLM names.
    judge_to_judge_agreement = sorted_dict_of_dict(judge_to_judge_agreement)
    judge_to_judge_agreement_df = pd.DataFrame(judge_to_judge_agreement)
    return judge_to_judge_agreement_df


def get_judge_agreement_map(df) -> dict[str, pd.DataFrame]:
    judge_agreement_dfs = {}
    for agreement_method in AGREEMENT_METHODS:
        judge_agreement_df = get_judge_agreement_df(df, agreement_method)
        judge_agreement_dfs[agreement_method] = judge_agreement_df
    return judge_agreement_dfs