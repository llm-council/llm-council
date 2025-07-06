import numpy as np
import pandas as pd

from lm_council.analysis.visualization import plot_heatmap


def get_judge_agreement(judging_df, eval_config):
    # Identify judge models and (example_id_column, model_being_judged) pairs
    judge_models = judging_df["judge_model"].unique()

    example_id_col = "user_prompt"
    model_col = "model_being_judged"
    criteria_names = [c.name for c in eval_config.config.rubric] + ["Overall"]

    agreement_matrices = {}

    for crit_col in criteria_names:
        # Build matrix: rows/cols = judge_models, values = mean absolute difference
        matrix = pd.DataFrame(index=judge_models, columns=judge_models, dtype=float)
        for jm1 in judge_models:
            for jm2 in judge_models:
                if jm1 == jm2:
                    # Skip self-comparison
                    matrix.loc[jm1, jm2] = float("nan")
                    continue
                # For each (example_id, model_being_judged), get scores from both judges
                merged = judging_df[
                    [example_id_col, model_col, "judge_model", crit_col]
                ].copy()
                merged = merged.rename(columns={crit_col: "score"})
                jm1_scores = merged[merged["judge_model"] == jm1][
                    [example_id_col, model_col, "score"]
                ]
                jm2_scores = merged[merged["judge_model"] == jm2][
                    [example_id_col, model_col, "score"]
                ]
                merged_scores = pd.merge(
                    jm1_scores,
                    jm2_scores,
                    on=[example_id_col, model_col],
                    suffixes=("_jm1", "_jm2"),
                )
                if not merged_scores.empty:
                    # Mean absolute percentage difference.
                    abs_diff = (
                        (merged_scores["score_jm1"] - merged_scores["score_jm2"]).abs()
                        / merged_scores["score_jm2"].replace(0, np.nan).abs()
                    ).mean()
                else:
                    abs_diff = float("nan")
                matrix.loc[jm1, jm2] = abs_diff

        agreement_matrices[crit_col] = matrix

    # Compute mean agreement for each criterion and model (including council)
    mean_agreement_rows = []
    for crit_col, matrix in agreement_matrices.items():
        # For each judge model (row), compute mean agreement with others (excluding self)
        for jm in matrix.index:
            # Exclude diagonal (self-comparison)
            mask = matrix.index != jm
            values = matrix.loc[jm, mask].values.astype(float)
            # Agreement is 1 - mean absolute percentage difference
            agreement = 1 - (np.nanmean(values) if len(values) > 0 else float("nan"))
            mean_agreement_rows.append(
                {"model": jm, "criterion": crit_col, "agreement": agreement}
            )
        # Council: mean agreement across all judge pairs (excluding diagonal)
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        council_values = matrix.values[mask].astype(float)
        council_agreement = (
            eval_config.config.prebuilt_likert_scale
            - (np.nanmean(council_values) if len(council_values) > 0 else float("nan"))
        ) / eval_config.config.prebuilt_likert_scale
        mean_agreement_rows.append(
            {
                "model": "council",
                "criterion": crit_col,
                "agreement": council_agreement,
            }
        )
    mean_agreement_df = pd.DataFrame(mean_agreement_rows)

    return agreement_matrices, mean_agreement_df
