import pandas as pd


def get_affinity_matrices(
    judging_df: pd.DataFrame, eval_config
) -> dict[str, pd.DataFrame]:
    judge_models = list(judging_df["judge_model"].unique())
    models_being_judged = list(judging_df["model_being_judged"].unique())
    criteria_names = [c.name for c in eval_config.config.rubric] + ["Overall"]

    affinity_matrices = {}

    for crit in criteria_names:
        # Build matrix: rows = models_being_judged, cols = judge_models + ["council"]
        matrix = pd.DataFrame(
            index=models_being_judged,
            columns=judge_models + ["council"],
            dtype=float,
        )
        for model in models_being_judged:
            # For each judge model
            for jm in judge_models:
                scores = judging_df[
                    (judging_df["model_being_judged"] == model)
                    & (judging_df["judge_model"] == jm)
                ][crit]
                matrix.loc[model, jm] = (
                    scores.mean() if not scores.empty else float("nan")
                )
            # Council column: mean across all judges
            all_scores = judging_df[judging_df["model_being_judged"] == model][crit]
            matrix.loc[model, "council"] = (
                all_scores.mean() if not all_scores.empty else float("nan")
            )

        affinity_matrices[crit] = matrix

    return affinity_matrices
