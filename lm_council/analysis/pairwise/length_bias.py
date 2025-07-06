import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from lm_council.analysis.pairwise.pairwise_utils import get_council_choice
from lm_council.constants import MAJOR_A_WIN, MAJOR_B_WIN, MINOR_A_WIN, MINOR_B_WIN, TIE


def get_num_words(text):
    """Returns the number of words in English."""
    return len(text.split())


def get_respondent_token_counts(
    respondents_df,
    example_id_column="emobench_id",
    llm_text_column="response_string",
    llm_responder_column="llm_responder",
) -> tuple[pd.DataFrame, dict[tuple[int, str], int]]:
    """Returns respondent -> list of word counts and (id, respondent) -> word count."""
    # Build a map from (id, completer) -> # tokens
    id_completer_to_num_words = {}
    completer_to_num_words = defaultdict(list)
    for i, row in respondents_df.iterrows():
        num_words = get_num_words(row[llm_text_column])
        id_completer_to_num_words[
            (row[example_id_column], row[llm_responder_column])
        ] = num_words
        completer_to_num_words[row[llm_responder_column]].append(num_words)

    completer_token_counts = pd.DataFrame(completer_to_num_words)
    completer_token_counts.columns = completer_token_counts.columns.str.split("/").str[
        -1
    ]
    return completer_token_counts, id_completer_to_num_words


def attach_num_words_to_judging_df(
    judging_df, id_respondent_to_num_words, example_id_column="emobench_id"
):
    # Find relationship between wins and token count differences
    def get_num_words_for_first_completion_by_row(row):
        return id_respondent_to_num_words[
            (row[example_id_column], row["first_completion_by"])
        ]

    def get_num_words_for_second_completion_by_row(row):
        return id_respondent_to_num_words[
            (row[example_id_column], row["second_completion_by"])
        ]

    # Attach information about the number of tokens.
    judging_df.loc[:, "first_completion_by_num_words"] = judging_df.apply(
        get_num_words_for_first_completion_by_row, axis=1
    )
    judging_df.loc[:, "second_completion_by_num_words"] = judging_df.apply(
        get_num_words_for_second_completion_by_row, axis=1
    )
    return judging_df


def get_length_bias(
    judging_df,
    id_respondent_to_num_words,
    name,
    outdir,
    example_id_column="emobench_id",
):
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

    judging_df = attach_num_words_to_judging_df(
        judging_df, id_respondent_to_num_words, example_id_column
    )

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


def get_length_biases_df(
    judging_df, id_respondent_to_num_words, example_id_column
) -> pd.DataFrame:
    """Uses linear regression to calculate the length bias for each judge and council."""
    length_biases = {}

    # Everyone.
    length_biases["council/no-aggregation"] = get_length_bias(
        judging_df,
        id_respondent_to_num_words,
        "council/no-aggregation",
        "logdir",
        example_id_column=example_id_column,
    )

    # With majority vote.
    council_choice_majority = get_council_choice(
        judging_df, "majority", example_id_column=example_id_column
    )
    length_biases["council/majority-vote"] = get_length_bias(
        council_choice_majority,
        id_respondent_to_num_words,
        "council/majority-vote",
        "logdir",
        example_id_column=example_id_column,
    )

    # With mean pooling.
    council_choice_mean_pooling = get_council_choice(
        judging_df, "mean_pooling", example_id_column=example_id_column
    )
    length_biases["council/mean-pooling"] = get_length_bias(
        council_choice_mean_pooling,
        id_respondent_to_num_words,
        "council/mean-pooling",
        "logdir",
        example_id_column=example_id_column,
    )

    # For individual judges.
    for judge in judging_df["judge_model"].unique():
        judge_choice = judging_df[judging_df["judge_model"] == judge]
        length_biases[judge] = get_length_bias(
            judge_choice,
            id_respondent_to_num_words,
            judge,
            "logdir",
            example_id_column=example_id_column,
        )

    return pd.DataFrame(length_biases)


def plot_length_based_outcomes(
    judging_df,
    name,
    id_respondent_to_num_words,
    show=True,
    outdir=None,
    example_id_column="emobench_id",
):
    """Creates a plot with X and Y axes as the response lengths of the battlers, and the color
    indicating the result of the battle.

    """
    # Attach num words to judging_df.
    attach_num_words_to_judging_df(
        judging_df, id_respondent_to_num_words, example_id_column
    )

    # Apply colors.
    def get_color(row):
        if row["pairwise_choice"] == MINOR_A_WIN:
            return "teal"
        if row["pairwise_choice"] == MINOR_B_WIN:
            return "orange"
        if row["pairwise_choice"] == MAJOR_A_WIN:
            return "blue"
        if row["pairwise_choice"] == MAJOR_B_WIN:
            return "red"
        return "yellow"

    judging_df.loc[:, "color"] = judging_df.apply(get_color, axis=1)

    # Filter out ties, as they cause a lot of noise.
    filtered_df = judging_df[judging_df["color"] != "yellow"]

    plt.figure(figsize=(10, 10))
    plt.scatter(
        filtered_df["second_completion_by_num_words"],  # X-axis data
        filtered_df["first_completion_by_num_words"],  # Y-axis data
        color=filtered_df["color"],  # Color by the 'color' column
        alpha=0.1,  # Set transparency
    )

    max_value = max(
        filtered_df["first_completion_by_num_words"].max(),
        filtered_df["second_completion_by_num_words"].max(),
    )
    min_value = min(
        filtered_df["first_completion_by_num_words"].min(),
        filtered_df["second_completion_by_num_words"].min(),
    )
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "k--",
        label="Equal # tokens",
    )  # 'k--' specifies a black dashed line

    plt.title(f"LLM Judge Outcomes ({name})")
    plt.xlabel("# Words (Respondent B)")
    plt.ylabel("# Words (Respondent A)")
    plt.grid(True)

    # Add custom legend for colors
    legend_handles = [
        mpatches.Patch(color="blue", label="Major A Win"),
        mpatches.Patch(color="teal", label="Minor A Win"),
        mpatches.Patch(color="orange", label="Minor B Win"),
        mpatches.Patch(color="red", label="Major B Win"),
    ]
    plt.legend(handles=legend_handles)

    plt.tight_layout()

    if outdir:
        plt.savefig(os.path.join(outdir, f"{name}.pdf"))

    if show:
        plt.show()
    plt.close()
