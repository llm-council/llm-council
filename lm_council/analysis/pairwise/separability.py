import math
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from lm_council.analysis.pairwise.agreement import get_side
from lm_council.analysis.pairwise.pairwise_utils import get_council_choice
from lm_council.constants import MAJOR_A_WIN, MAJOR_B_WIN, MINOR_A_WIN, MINOR_B_WIN, TIE


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    # Modified from https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {a: [wins[a][b] if a != b else np.nan for b in names] for a in names}

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(df, column, baseline):
    # Modified from https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_win_rate(bootstrap_online_elo, reference_llm_respondent):
    # Modified from https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py
    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats["results"].astype("object")

    for i, model in enumerate(bootstrap_online_elo.index):
        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]

    return get_win_rate_column(stats, "score", reference_llm_respondent)


def convert_judging_df_to_arena_hard_battles_df(judging_df, WEIGHT=3):
    """Returns a distilled dataframe of battle results.

    The returned dataframe has the following columns:
        - question_id
        - model_a
        - model_b
        - winner

    The reason why we do this is so that more of Arena Hard's ELO computation library can be used
    out of the box.
    """
    battles = []
    for _, row in judging_df.iterrows():
        output = {
            # "question_id": row[example_id_column],
            "model_a": row["first_completion_by"],
            "model_b": row["second_completion_by"],
        }

        if row["pairwise_choice"] in {MAJOR_A_WIN, MAJOR_B_WIN}:
            weight = WEIGHT
        else:
            weight = 1

        pairwise_choice_side = get_side(row["pairwise_choice"])
        if pairwise_choice_side == "A":
            output["winner"] = "model_a"
        elif pairwise_choice_side == "B":
            output["winner"] = "model_b"
        elif pairwise_choice_side == "Tie":
            output["winner"] = "tie"

        battles.extend([output] * weight)

    return pd.DataFrame(battles)


def compute_mle_elo(
    judging_df,
    reference_llm_respondent,
    SCALE=400,
    BASE=10,
    INIT_RATING=1000,
) -> pd.Series:
    """Returns a Series of ELO ratings with the anchor's reference_llm_respondent score pinned to INIT_RATING (default=1000)."""
    df = convert_judging_df_to_arena_hard_battles_df(judging_df)

    # Verbatim copy of https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # Counts 1 tie as 1 A win and 1 B win, which is why we duplicate the battles.
    # tie_idx = df["winner"] == "A=B"
    tie_idx = df["winner"] == "tie"
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as reference_llm_respondent = 1000
    if reference_llm_respondent in models.index:
        elo_scores += 1000 - elo_scores[models[reference_llm_respondent]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(
    df,
    func_compute_elo,
    num_rounds,
    reference_llm_respondent,
):
    rows = []
    for i in tqdm(range(num_rounds), desc="bootstrap"):
        if len(df) < 100:
            # If there are fewer than 100 rows, we cannot bootstrap.
            rows.append(func_compute_elo(df, reference_llm_respondent))
            continue

        rows.append(
            func_compute_elo(
                df.sample(n=len(df), replace=True),
                reference_llm_respondent,
            )
        )
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def get_stats_for_bootstrap(
    bootstrap_online_elo, bootstrap_elo_lu, reference_llm_respondent
):
    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats["results"].astype("object")

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)
        stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()

    stats.sort_values(by="model", inplace=True)
    stats["score"] = get_win_rate_column(
        stats, "score", reference_llm_respondent
    ).tolist()
    stats["lower"] = get_win_rate_column(
        stats, "lower", reference_llm_respondent
    ).tolist()
    stats["upper"] = get_win_rate_column(
        stats, "upper", reference_llm_respondent
    ).tolist()
    decimal = 1

    stats = stats.sort_values(by="score", ascending=False)
    stats["lower_ci"] = (stats["lower"] - stats["score"]).round(decimal)
    stats["upper_ci"] = (stats["upper"] - stats["score"]).round(decimal)
    stats["score_with_ci_str"] = (
        stats["score"].astype(str)
        + " ("
        + stats["lower_ci"].astype(str)
        + ", "
        + stats["upper_ci"].astype(str)
        + ")"
    )
    for _, row in stats.iterrows():
        interval = str(
            (
                round(row["lower"] - row["score"], decimal),
                round(row["upper"] - row["score"], decimal),
            )
        )
    return stats


def calculate_non_overlapping_percentage(
    df, model_column_name="model", lower_column_name="lower", upper_column_name="upper"
):
    # Extract the model names and intervals
    models = df[model_column_name]
    intervals = df[[lower_column_name, upper_column_name]]

    # Calculate all possible pairs of models
    total_pairs = 0
    non_overlapping_pairs = 0

    for (model1, interval1), (model2, interval2) in combinations(
        zip(models, intervals.itertuples(index=False, name=None)), 2
    ):
        total_pairs += 1
        lower1, upper1 = interval1
        lower2, upper2 = interval2

        # Check if intervals do not overlap
        if upper1 < lower2 or upper2 < lower1:
            non_overlapping_pairs += 1

    # Calculate the percentage of non-overlapping pairs
    if total_pairs > 0:
        percentage = (non_overlapping_pairs / total_pairs) * 100
    else:
        percentage = 0  # In case there are not enough models to form a pair

    return percentage


def filter_ratings_by_allowlist(df, judge_model_allowlist):
    return df[df["judge_model"].isin(judge_model_allowlist)]


def get_elo_rankings(judging_df, reference_llm_respondent, bootstrap_rounds):
    """Returns the ELO rankings for the given judging_df using Bradley-Terry and bootstrapped CIs."""
    bootstrap_online_elo = compute_mle_elo(judging_df, reference_llm_respondent)
    bootstrap_elo_lu = get_bootstrap_result(
        judging_df,
        compute_mle_elo,
        bootstrap_rounds,
        reference_llm_respondent,
    )
    elo_scores = get_stats_for_bootstrap(
        bootstrap_online_elo, bootstrap_elo_lu, reference_llm_respondent
    )
    separability = calculate_non_overlapping_percentage(elo_scores)
    return {
        "elo_scores": elo_scores,
        "separability": separability,
        "polarization": elo_scores["score"].max() - elo_scores["score"].min(),
    }


def analyze_rankings_separability_polarization(
    judging_df,
    reference_llm_respondent,
    bootstrap_rounds,
    include_individual_judges=False,
    include_council_majority=True,
    include_council_mean_pooling=False,
    include_council_no_aggregation=False,
    example_id_column="emobench_id",
) -> dict:
    """Produces rankings based on ELO scores, and uses that to compute separability and polarization."""
    # Map of judge to its ELO rankings.
    judge_to_elo_rankings: dict[str, pd.DataFrame] = {}

    # Everyone's vote.
    if include_council_no_aggregation:
        judge_to_elo_rankings["council/no-aggregation"] = get_elo_rankings(
            judging_df, reference_llm_respondent, bootstrap_rounds
        )

    # Council, with majority aggregation.
    if include_council_majority:
        council_choice = get_council_choice(judging_df, "majority", example_id_column)
        judge_to_elo_rankings["council/majority-vote"] = get_elo_rankings(
            council_choice, reference_llm_respondent, bootstrap_rounds
        )

    # Council, with mean pooling aggregation.
    if include_council_mean_pooling:
        council_choice = get_council_choice(
            judging_df, "mean_pooling", example_id_column
        )
        judge_to_elo_rankings["council/mean-pooling"] = get_elo_rankings(
            council_choice, reference_llm_respondent, bootstrap_rounds
        )

    # Individual council members.
    if include_individual_judges:
        judge_models = list(judging_df["judge_model"].unique())
        for judge_model in judge_models:
            judge_choice = filter_ratings_by_allowlist(judging_df, [judge_model])
            judge_to_elo_rankings[judge_model] = get_elo_rankings(
                judge_choice, reference_llm_respondent, bootstrap_rounds
            )

    return judge_to_elo_rankings
