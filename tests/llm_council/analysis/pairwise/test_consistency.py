import pytest
import pandas as pd
from collections import defaultdict

from llm_council.analysis.pairwise.consistency import (
    get_consistent_votes,
    consistency_with_sidewise_tolerance_fn,
    consistency_strict_fn,
)
from llm_council.constants import MAJOR_A_WIN, MINOR_B_WIN, TIE


@pytest.fixture
def mock_df():
    data = [
        {
            "llm_judge": "judge1",
            "emobench_id": "example1",
            "first_completion_by": "modelA",
            "second_completion_by": "modelB",
            "pairwise_choice": MAJOR_A_WIN,
        },
        {
            "llm_judge": "judge1",
            "emobench_id": "example1",
            "first_completion_by": "modelB",
            "second_completion_by": "modelA",
            "pairwise_choice": MINOR_B_WIN,
        },
        {
            "llm_judge": "judge2",
            "emobench_id": "example2",
            "first_completion_by": "modelC",
            "second_completion_by": "modelD",
            "pairwise_choice": TIE,
        },
        {
            "llm_judge": "judge2",
            "emobench_id": "example2",
            "first_completion_by": "modelD",
            "second_completion_by": "modelC",
            "pairwise_choice": TIE,
        },
    ]
    return pd.DataFrame(data)


def test_sidewise_consistency(mock_df):
    """Test for sidewise consistency function"""
    result = get_consistent_votes(mock_df, consistency_fn_name="sidewise")
    assert len(result) == 4


def test_strict_consistency(mock_df):
    """Test for strict consistency function"""
    result = get_consistent_votes(mock_df, consistency_fn_name="strict")
    assert len(result) == 2


def test_no_consistent_votes_strict(mock_df):
    """Test case where there are no consistent votes with strict consistency"""
    inconsistent_df = pd.DataFrame(
        [
            {
                "llm_judge": "judge1",
                "emobench_id": "example1",
                "first_completion_by": "modelA",
                "second_completion_by": "modelB",
                "pairwise_choice": "MAJOR_A_WIN",
            },
            {
                "llm_judge": "judge1",
                "emobench_id": "example1",
                "first_completion_by": "modelB",
                "second_completion_by": "modelA",
                "pairwise_choice": "MAJOR_A_WIN",
            },
        ]
    )
    result = get_consistent_votes(inconsistent_df, consistency_fn_name="strict")
    assert len(result) == 0


def test_get_consistent_votes_edge_case():
    """Test with minimal or empty dataframe"""
    empty_df = pd.DataFrame(
        columns=[
            "llm_judge",
            "emobench_id",
            "first_completion_by",
            "second_completion_by",
            "pairwise_choice",
        ]
    )
    result = get_consistent_votes(empty_df, consistency_fn_name="strict")
    assert result.empty


def test_consistency_with_testdata():
    df = pd.read_json("tests/testdata/lmc_ei.jsonl", lines=True, orient="records")
    result = get_consistent_votes(df, consistency_fn_name="strict")
    assert len(result) == 992

    result = get_consistent_votes(df, consistency_fn_name="sidewise")
    assert len(result) == 994
