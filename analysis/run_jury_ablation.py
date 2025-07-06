import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import math
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from datasets import load_dataset
import itertools
import random
from itertools import combinations
import pickle as pkl


def calculate_non_overlapping_percentage(
    df, model_column_name="model", lower_column_name="lower", upper_column_name="upper"
):
    """Returns the percentage of non-overlapping intervals."""
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


def get_win_rate_column(df, column, baseline):
    """Uses Terry-Bradley to get the predicted win rate from elo stats for a specific model."""
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_win_rate(bootstrap_online_elo, reference_llm_completer):
    """Uses Terry-Bradley to get the predicted win rate from elo stats for all models."""
    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats["results"].astype("object")

    for i, model in enumerate(bootstrap_online_elo.index):
        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]

    return get_win_rate_column(stats, "score", reference_llm_completer)


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    """Predicts the win rate given ELO ratings, a specified scale, base, and init rating."""
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {a: [wins[a][b] if a != b else np.NAN for b in names] for a in names}

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_battles_from_judgment(df, example_id_column="emobench_id", WEIGHT=3):
    """Maps battle outcomes to specific winners. The WEIGHT is used to weight strong votes."""
    # Modified from https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py#L112C1-L176C30
    battles = []
    for _, row in df.iterrows():
        output = {
            "question_id": row[example_id_column],
            "model_a": row["first_completion_by"],
            "model_b": row["second_completion_by"],
        }

        weight = 1
        if row["pairwise_choice"] == "A=B":
            output["winner"] = "tie"
        elif row["pairwise_choice"] == "A>B":
            output["winner"] = "model_a"
        elif row["pairwise_choice"] == "A>>B":
            output["winner"] = "model_a"
            weight = WEIGHT
        elif row["pairwise_choice"] == "B>A":
            output["winner"] = "model_b"
        elif row["pairwise_choice"] == "B>>A":
            output["winner"] = "model_b"
            weight = WEIGHT
        else:
            print("Unknown pairwise_choice: " + row["pairwise_choice"])

        for i in range(weight):
            battles.append(output)

    return pd.DataFrame(battles)


def compute_mle_elo(
    df, reference_llm_completer=None, SCALE=400, BASE=10, INIT_RATING=1000
):
    """Compute the ELO scores for all models."""
    df = get_battles_from_judgment(df)

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
    tie_idx = df["winner"] == "tie"
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as reference_llm_completer = 1000
    if reference_llm_completer and reference_llm_completer in models.index:
        elo_scores += 1000 - elo_scores[models[reference_llm_completer]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def sample_combinations(items, choose, instances):
    """Generate all possible combinations of the specified size, and sample them randomly."""
    # This is actually fully determinstic for n choose n.
    return [random.choices(items, k=choose) for _ in range(instances)]


def sample_llm_council_members(llm_council_members, k, n):
    """
    Generates a list of n instances, each being a random sample of size k
    from lm_council_members with replacement.

    Parameters:
    llm_council_members (list): The list of council members to sample from.
    k (int): The size of each sample.
    n (int): The number of instances to generate.

    Returns:
    list: A list containing n samples, each of size k.
    """
    return [random.choices(llm_council_members, k=k) for _ in range(n)]


def filter_ratings_by_allowlist(df, llm_judge_allowlist):
    """Filter ratings by a list of allowed judges."""
    return df[df["judge_model"].isin(llm_judge_allowlist)]


def get_relevant_ratings_for_trial(df, emo_bench_ids, llm_judges):
    # There can be repeated members in both cases, assuming we are sampling with replacement.
    # First get a map of (emo_bench_id, llm_judge) -> relevant ratings.
    emo_bench_id_and_judge_to_ratings_map = defaultdict(dict)
    for emo_bench_id in set(emo_bench_ids):
        for llm_judge in set(llm_judges):
            df_subset = df[df["emobench_id"] == emo_bench_id]
            emo_bench_id_and_judge_to_ratings_map[emo_bench_id][llm_judge] = (
                filter_ratings_by_allowlist(df_subset, [llm_judge])
            )

    # Once the map is prepared, use pd.concat to merge all ratings.
    combined_df = pd.DataFrame()
    for emo_bench_id in emo_bench_ids:
        for llm_judge in llm_judges:
            ratings_for_this_emobench_id_and_llm_judge = (
                emo_bench_id_and_judge_to_ratings_map[emo_bench_id][llm_judge]
            )
            combined_df = pd.concat(
                [combined_df, ratings_for_this_emobench_id_and_llm_judge]
            )

    return combined_df


def get_adversarial_votes(df, num_adversarial_judges):
    """Get a dataframe of adversarial judge ratings (totally random)."""
    all_dfs = [get_adversarial_judge_ratings(df) for i in range(num_adversarial_judges)]
    return pd.concat(all_dfs)


def get_adversarial_judge_ratings(df):
    """Get ratings for adversarial judges."""
    # Extract unique combinations of 'id', 'first_completion_by', 'second_completion_by'
    adversarial_judge_df = df[
        ["emobench_id", "first_completion_by", "second_completion_by"]
    ].drop_duplicates()

    # Assign fixed values
    adversarial_judge_df["judge_model"] = "adversarial"
    adversarial_judge_df["metadata"] = None

    # Assign random pairwise choices.
    adversarial_judge_df["pairwise_choice"] = np.random.choice(
        ["A>B", "B>A", "A>>B", "B>>A"], size=len(adversarial_judge_df)
    )
    return adversarial_judge_df


def get_simulated_leaderboard_with_adversarial_judges(
    df,
    unique_council_members,
    jury_size,
    num_trials,
    num_adversarial_judges,
    num_adversarial_judges_frac,
    adversarial_method,
    simulation_normalized_number_judges,
    reference_llm_completer,
    num_emobench_ids,
):
    """Get a simulated leaderboard with adversarial judges."""
    win_rate_dict = defaultdict(list)
    rank_dict = defaultdict(list)
    elo_dict = defaultdict(list)
    sampled_jury_compositions = sample_combinations(
        unique_council_members, jury_size, num_trials
    )

    council_win_rate_dict = defaultdict(list)
    council_rank_dict = defaultdict(list)
    council_elo_dict = defaultdict(list)

    # Determine the number of adversarial judges.
    num_real_judges = len(sampled_jury_compositions[0])
    if num_adversarial_judges_frac:
        # 1, 0.5 -> 2, 5, 0.5 -> 7
        num_judges_including_adversaries = int(
            num_real_judges / num_adversarial_judges_frac
        )
        num_adversarial_judges = num_judges_including_adversaries - num_real_judges
    elif num_adversarial_judges:
        num_judges_including_adversaries = num_real_judges + num_adversarial_judges
    else:
        num_judges_including_adversaries = num_real_judges
        num_adversarial_judges = 0
    print(f"- Number of judges: {num_real_judges}")
    print(f"- Number of adversarial judges: {num_adversarial_judges}")
    print(
        f"- Number of judges including adversaries: {num_judges_including_adversaries}"
    )

    for sampled_jury_composition in sampled_jury_compositions:
        sampled_emobench_ids = sample_combinations(
            df["emobench_id"].unique(), num_emobench_ids, 1
        )[0]
        relevant_ratings = get_relevant_ratings_for_trial(
            df, sampled_emobench_ids, sampled_jury_composition
        )

        # Add adversarial judges, if applicable.
        if num_adversarial_judges:
            adversarial_df = get_adversarial_votes(
                relevant_ratings, num_adversarial_judges, adversarial_method
            )
            relevant_ratings = pd.concat([adversarial_df, relevant_ratings])

        # Apply council majority aggregation.
        # This step is important in order to reduce the number of total battles for bootstrapping.
        relevant_ratings = (
            relevant_ratings.groupby(
                ["emobench_id", "first_completion_by", "second_completion_by"]
            )["pairwise_choice"]
            .agg(lambda x: x.mode()[0])
            .reset_index()
        )

        # Compute the ELO and bootstrap them.
        bootstrap_online_elo = compute_mle_elo(
            relevant_ratings, reference_llm_completer
        )

        # Save scores and win rates.
        for llm_completer, elo_score in bootstrap_online_elo.to_dict().items():
            elo_dict[llm_completer].append(elo_score)

        win_rates = get_win_rate(bootstrap_online_elo, reference_llm_completer)
        for llm_completer, win_rate in win_rates.to_dict().items():
            win_rate_dict[llm_completer].append(win_rate)

        ranks = bootstrap_online_elo.rank(method="min", ascending=False)
        for llm_completer, rank in ranks.items():
            rank_dict[llm_completer].append(rank)

    stats = pd.DataFrame()
    for i, model in enumerate(bootstrap_online_elo.index):
        stats.at[i, "model"] = model
        stats.at[i, "rank_score"] = np.mean(rank_dict[model])
        stats.at[i, "rank_std"] = np.std(rank_dict[model])
        stats.at[i, "rank_lower"] = np.percentile(rank_dict[model], 2.5)
        stats.at[i, "rank_upper"] = np.percentile(rank_dict[model], 97.5)

        stats.at[i, "elo_score"] = np.mean(elo_dict[model])
        stats.at[i, "elo_std"] = np.std(elo_dict[model])
        stats.at[i, "elo_lower"] = np.percentile(elo_dict[model], 2.5)
        stats.at[i, "elo_upper"] = np.percentile(elo_dict[model], 97.5)

        stats.at[i, "win_rate_score"] = np.mean(win_rate_dict[model])
        stats.at[i, "win_rate_std"] = np.std(win_rate_dict[model])
        stats.at[i, "win_rate_lower"] = np.percentile(win_rate_dict[model], 2.5)
        stats.at[i, "win_rate_upper"] = np.percentile(win_rate_dict[model], 97.5)

    stats["separability"] = calculate_non_overlapping_percentage(
        stats,
        model_column_name="model",
        lower_column_name="win_rate_lower",
        upper_column_name="win_rate_upper",
    )
    return stats


def get_jury_ablation_stats(
    df,
    num_trials,
    unique_council_members,
    num_adversarial_judges,
    num_adversarial_judges_frac,
    adversarial_method,
    reference_llm_completer,
    num_emobench_ids,
):
    """Get a simulated leaderboard with number of adversarial judges."""
    jury_ablation_stats_with_adversarial_judges = pd.DataFrame()

    # Map None to 0.
    if num_adversarial_judges is None:
        num_adversarial_judges = 0
    if num_adversarial_judges_frac is None:
        num_adversarial_judges_frac = 0

    if num_adversarial_judges:
        simulation_normalized_number_judges = (
            len(unique_council_members) + num_adversarial_judges
        )
    elif num_adversarial_judges_frac:
        simulation_normalized_number_judges = int(
            len(unique_council_members) / num_adversarial_judges_frac
        )
    else:
        simulation_normalized_number_judges = len(unique_council_members)
    print(
        f"Number of situation-normalized council members: {simulation_normalized_number_judges}"
    )

    for jury_size in range(1, len(unique_council_members) + 1):
        print(f"Evaluating jury size: {jury_size}")
        num_retries_left = 3
        while num_retries_left > 0:
            try:
                stats = get_simulated_leaderboard_with_adversarial_judges(
                    df,
                    unique_council_members,
                    jury_size,
                    num_trials,
                    num_adversarial_judges,
                    num_adversarial_judges_frac,
                    adversarial_method,
                    simulation_normalized_number_judges,
                    reference_llm_completer,
                    num_emobench_ids,
                )
                break
            except Exception as e:
                print(f"Encountered separability issue: {e}")
                import traceback

                traceback.print_exc()
                num_retries_left -= 1

        jury_ablation_stats_with_adversarial_judges.at[jury_size, "rank_std.mean"] = (
            stats["rank_std"].mean()
        )
        print(stats["rank_std"].mean())
        jury_ablation_stats_with_adversarial_judges.at[jury_size, "rank_std.std"] = (
            stats["rank_std"].std()
        )
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "rank_std.ci.upper"
        ] = np.percentile(stats["rank_std"], 97.5)
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "rank_std.ci.lower"
        ] = np.percentile(stats["rank_std"], 2.5)
        jury_ablation_stats_with_adversarial_judges.at[jury_size, "elo_std.mean"] = (
            stats["elo_std"].mean()
        )
        jury_ablation_stats_with_adversarial_judges.at[jury_size, "elo_std.std"] = (
            stats["elo_std"].std()
        )
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "elo_std.ci.upper"
        ] = np.percentile(stats["elo_std"], 97.5)
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "elo_std.ci.lower"
        ] = np.percentile(stats["elo_std"], 2.5)
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "win_rate_std.mean"
        ] = stats["win_rate_std"].mean()
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "win_rate_std.std"
        ] = stats["win_rate_std"].std()
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "win_rate_std.ci.upper"
        ] = np.percentile(stats["win_rate_std"], 97.5)
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "win_rate_std.ci.lower"
        ] = np.percentile(stats["win_rate_std"], 2.5)

        # Separability
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "separability.mean"
        ] = stats["separability"].mean()
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "separability.std"
        ] = stats["separability"].std()
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "separability.ci.upper"
        ] = np.percentile(stats["separability"], 97.5)
        jury_ablation_stats_with_adversarial_judges.at[
            jury_size, "separability.ci.lower"
        ] = np.percentile(stats["separability"], 2.5)

    print("Finished jury ablation.")
    return jury_ablation_stats_with_adversarial_judges


# Read the data.
dataset = load_dataset("llm-council/emotional_application", "response_judging")
df = dataset["council"].to_pandas()

# Example of one jury ablation, without any adversarial judges.

# Ablate over number of emobench ids.
all_jury_ablation_stats = []
for num_emobench_ids in tqdm(range(10, 101, 10), desc="Processing Ablation Trials"):
    jury_ablation_stats = get_jury_ablation_stats(
        df=df,
        unique_council_members=df["judge_model"].unique(),
        num_trials=100,
        num_adversarial_judges=None,
        num_adversarial_judges_frac=None,
        adversarial_method=None,
        reference_llm_completer="qwen1.5-32B-Chat",
        num_emobench_ids=num_emobench_ids,
    )
    all_jury_ablation_stats.append(jury_ablation_stats)


with open("all_ablation_data.pkl", "wb") as fp:
    pkl.dump(all_jury_ablation_stats, fp)
