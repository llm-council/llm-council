from collections import defaultdict
from llm_council.constants import (
    MAJOR_A_WIN,
    MINOR_A_WIN,
    MINOR_B_WIN,
    MAJOR_B_WIN,
    TIE,
)
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from llm_council.analysis.pairwise.pairwise_utils import get_council_choice


def get_num_words(text):
    """Returns the number of words in English."""
    return len(text.split())


def get_respondent_token_counts(
    respondents_df,
) -> tuple[pd.DataFrame, dict[tuple[int, str], int]]:
    """Returns respondent -> list of word counts and (id, respondent) -> word count."""
    # Build a map from (id, completer) -> # tokens
    id_completer_to_num_words = {}
    completer_to_num_words = defaultdict(list)
    for i, row in respondents_df.iterrows():
        num_words = get_num_words(row["response_string"])
        id_completer_to_num_words[(row["emobench_id"], row["llm_responder"])] = (
            num_words
        )
        completer_to_num_words[row["llm_responder"]].append(num_words)

    completer_token_counts = pd.DataFrame(completer_to_num_words)
    completer_token_counts.columns = completer_token_counts.columns.str.split("/").str[
        -1
    ]
    return completer_token_counts, id_completer_to_num_words


def attach_num_words_to_judging_df(judging_df, id_respondent_to_num_words):
    # Find relationship between wins and token count differences
    def get_num_words_for_first_completion_by_row(row):
        return id_respondent_to_num_words[
            (row["emobench_id"], row["first_completion_by"])
        ]

    def get_num_words_for_second_completion_by_row(row):
        return id_respondent_to_num_words[
            (row["emobench_id"], row["second_completion_by"])
        ]

    # Attach information about the number of tokens.
    judging_df.loc[:, "first_completion_by_num_words"] = judging_df.apply(
        get_num_words_for_first_completion_by_row, axis=1
    )
    judging_df.loc[:, "second_completion_by_num_words"] = judging_df.apply(
        get_num_words_for_second_completion_by_row, axis=1
    )
    return judging_df


def get_length_bias(judging_df, id_respondent_to_num_words, name, outdir):
    def get_result_numeric(row):
        if (
            row["pairwise_choice"] == MINOR_A_WIN
            or row["pairwise_choice"] == MAJOR_A_WIN
        ):
            return 1
        if (
            row["pairwise_choice"] == MINOR_B_WIN
            or row["pairwise_choice"] == MAJOR_B_WIN
        ):
            return -1
        return 0

    judging_df = attach_num_words_to_judging_df(judging_df, id_respondent_to_num_words)

    # Calculate the difference in tokens
    judging_df.loc[:, "token_diff"] = (
        judging_df["first_completion_by_num_words"]
        - judging_df["second_completion_by_num_words"]
    )
    judging_df.loc[:, "result_numeric"] = judging_df.apply(get_result_numeric, axis=1)
    X = judging_df[["token_diff"]]
    y = judging_df["result_numeric"]
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)

    return {
        "length_bias": r_squared,
    }


def get_length_biases_df(judging_df, id_respondent_to_num_words) -> pd.DataFrame:
    """Uses linear regression to calculate the length bias for each judge and council."""
    length_biases = {}

    # Everyone.
    length_biases["council (no aggregation)"] = get_length_bias(
        judging_df, id_respondent_to_num_words, "council (no aggregation)", "logdir"
    )

    # With majority vote.
    council_choice_majority = get_council_choice(judging_df, "majority")
    length_biases["council (majority vote)"] = get_length_bias(
        council_choice_majority,
        id_respondent_to_num_words,
        "council (majority vote)",
        "logdir",
    )

    # With mean pooling.
    council_choice_mean_pooling = get_council_choice(judging_df, "mean_pooling")
    length_biases["council (mean pooling)"] = get_length_bias(
        council_choice_mean_pooling,
        id_respondent_to_num_words,
        "council (mean pooling)",
        "logdir",
    )

    # For individual judges.
    for judge in judging_df["llm_judge"].unique():
        judge_choice = judging_df[judging_df["llm_judge"] == judge]
        length_biases[judge] = get_length_bias(
            judge_choice, id_respondent_to_num_words, judge, "logdir"
        )

    return pd.DataFrame(length_biases)
