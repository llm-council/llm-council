import numpy as np
import pandas as pd

from lm_council.constants import MAJOR_A_WIN, MAJOR_B_WIN, MINOR_A_WIN, MINOR_B_WIN, TIE


def remove_self_judges(df):
    """Removes all examples where the LLM judged themselves."""
    df = df[df["judge_model"] != df["first_completion_by"]]
    df = df[df["judge_model"] != df["second_completion_by"]]
    return df


def get_mean_pooling_choice(agg_list):
    choice_to_value_map = {
        MAJOR_A_WIN: 2,
        MINOR_A_WIN: 1,
        TIE: 0,
        MINOR_B_WIN: -1,
        MAJOR_B_WIN: -2,
    }
    value_to_choice_map = {
        2: MAJOR_A_WIN,
        1: MINOR_A_WIN,
        0: TIE,
        -1: MINOR_B_WIN,
        -2: MAJOR_B_WIN,
    }
    numeric_values = agg_list.apply(lambda x: choice_to_value_map[x])
    return value_to_choice_map[round(np.mean(numeric_values))]


def get_council_choice(df, council_aggregation_method, example_id_column="emobench_id"):
    if council_aggregation_method == "majority":
        df_aggregated = (
            df.groupby(
                [example_id_column, "first_completion_by", "second_completion_by"]
            )
            .agg(
                pairwise_choice=pd.NamedAgg(
                    column="pairwise_choice", aggfunc=lambda x: x.mode()[0]
                )
            )
            .reset_index()
        )
        return df_aggregated
    if council_aggregation_method == "mean_pooling":
        # Council, by mean pooling.
        df_aggregated = (
            df.groupby(
                [example_id_column, "first_completion_by", "second_completion_by"]
            )
            .agg(
                pairwise_choice=pd.NamedAgg(
                    column="pairwise_choice", aggfunc=get_mean_pooling_choice
                )
            )
            .reset_index()
        )
        return df_aggregated
    raise ValueError(
        f"Invalid council aggregation method: {council_aggregation_method}"
    )


def get_reference_llm(judging_df, eval_config) -> str:
    """Get the reference LLM respondent for pairwise analysis."""

    if eval_config.config.algorithm_config.reference_models:
        # Use the reference LLM from the eval_config if defined.
        reference_llm_respondent = eval_config.config.algorithm_config.reference_models[
            0
        ]
        print(f"Using reference LLM from config: {reference_llm_respondent}")
    else:
        # Use the first judged model.
        reference_llm_respondent = judging_df.iloc[0]["first_completion_by"]
        print(f"Using reference LLM from judging_df: {reference_llm_respondent}")
    return reference_llm_respondent
