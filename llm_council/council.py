import pandas as pd
import requests
import os
import json
from llm_council.judging.config import EvaluationConfig
from llm_council.judging import PRESET_EVAL_CONFIGS
from llm_council.structured_outputs import create_dynamic_schema
import tqdm.asyncio
import random
from llm_council.judging.prompt_builder import (
    LIKERT_PREBUILT_MAP,
    check_prompt_template_contains_all_placeholders,
)
from llm_council.structured_outputs import (
    PAIRWISE_COMPARISON_LABEL_MAP,
    get_pairwise_comparison_schema,
)
from openai import AsyncOpenAI
import re
from llm_council.analysis.pairwise.bradley_terry import bradley_terry_analysis
from llm_council.analysis.visualization import plot_heatmap
from llm_council.analysis.visualization import (
    plot_arena_hard_elo_stats,
    plot_direct_assessment_charts,
)
from llm_council.analysis.pairwise.separability import (
    analyze_rankings_separability_polarization,
)
import aiohttp
from aiolimiter import AsyncLimiter
import instructor
from llm_council.analysis.pairwise.explicit_win_rate import get_explicit_win_rates
from llm_council.analysis.pairwise.agreement import get_judge_agreement_map
from llm_council.analysis.rubric.agreement import get_judge_agreement
from llm_council.analysis.rubric.affinity import get_affinity_matrices
from llm_council.analysis.pairwise.affinity import get_affinity_df
from llm_council.analysis.pairwise.pairwise_utils import get_reference_llm
import seaborn as sns
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, HfFolder
import numpy as np
import matplotlib.pyplot as plt


def process_pairwise_choice(raw_pairwise_choice: str) -> str:
    return raw_pairwise_choice.replace("[", "").replace("]", "")


class LanguageModelCouncil:

    def __init__(
        self,
        models: list[str],
        judge_models: list[str] | None = None,
        eval_config: EvaluationConfig | None = PRESET_EVAL_CONFIGS["default_rubric"],
    ):
        self.models = models
        self.eval_config = eval_config

        self.judge_models = judge_models
        # If no judge models are provided, use the same models for judging.
        if judge_models is None:
            # Use the same models for judging.
            self.judge_models = models

        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.client_structured = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.client_structured = instructor.from_openai(
            self.client_structured, mode=instructor.Mode.TOOLS
        )

        # ───── Fetch key-specific rate limits once ──────────
        key_meta = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        ).json()["data"]["rate_limit"]

        max_calls = key_meta["requests"]  # 100
        interval_seconds = (
            10
            if key_meta["interval"].endswith("s")
            else int(key_meta["interval"][:-1])  # handle "1m", "2h", etc.
        )
        self._limiter = AsyncLimiter(max_calls, interval_seconds)

        # If we're doing pairwise_comparisons with fixed_reference_model(s), check that each of the
        # models in config.algorithm_config.reference_models is also included in our completion
        # requests.
        if (
            self.eval_config.type == "pairwise_comparison"
            and getattr(self.eval_config.config, "algorithm_type", None)
            == "fixed_reference_models"
        ):
            reference_models = set(
                self.eval_config.config.algorithm_config.reference_models
            )
            missing = list(reference_models - set(self.models))

            print(
                f"The following reference models are specified in config.algorithm_config.reference_models but are not present in self.models: {missing}. Adding these model(s) to the council."
            )
            self.models.extend(missing)

        # List of all user prompts.
        self.user_prompts = []

        # List of all completions.
        self.completions = []

        # List of all judgments.
        self.judgments = []

    async def _with_rate_limit(self, coro):
        """Run any OpenRouter coroutine under the global AsyncLimiter."""
        async with self._limiter:
            return await coro

    async def get_judge_rubric_structured_output(
        self,
        user_prompt: str,
        judge_prompt: str,
        judge_model: str,
        model_being_judged: str,
        schema_class: type,
    ) -> dict:
        """Get an async structured output task for the given prompt."""
        structured_output, completion = await self._with_rate_limit(
            self.client_structured.chat.completions.create_with_completion(
                model=judge_model,
                messages=[
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=self.eval_config.temperature,
                response_model=schema_class,
                extra_body={"provider": {"require_parameters": True}},
            )
        )

        return {
            "user_prompt": user_prompt,
            "judge_model": judge_model,
            "model_being_judged": model_being_judged,
            "structured_output": structured_output,
            "temperature": self.eval_config.temperature,
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    async def get_text_completion(
        self,
        user_prompt: str,
        model: str,
        temperature: float | None = None,
    ):
        """Get an async text completion task for the given prompt."""
        completion = await self._with_rate_limit(
            self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
        )

        return {
            "user_prompt": user_prompt,
            "model": model,
            "completion_text": completion.choices[0].message.content,
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    async def get_text_completions(
        self,
        user_prompt: str,
        temperature: float | None = None,
    ):
        """Get an async text completion task for the given prompt."""
        return [
            self.get_text_completion(user_prompt, model, temperature)
            for model in self.models
        ]

    async def collect_completions(
        self, user_prompt: str, temperature: float | None = None
    ) -> pd.DataFrame:
        tasks = await self.get_text_completions(
            user_prompt,
            temperature=temperature,
        )

        print(f"Generated {len(tasks)} completion tasks for user prompt: {user_prompt}")

        completions = []
        for future in tqdm.asyncio.tqdm.as_completed(tasks, total=len(tasks)):
            completions.append(await future)

        return pd.DataFrame(completions)

    async def get_judge_rubric_tasks(
        self,
        completions_df: pd.DataFrame,
        temperature: float | None = None,
    ) -> list:
        judging_tasks = []
        for judge_model in self.judge_models:
            for _, row in completions_df.iterrows():
                model_being_judged = row["model"]

                # Skip if the judge model is the same as the model being judged.
                if (
                    self.eval_config.exclude_self_grading
                    and judge_model == model_being_judged
                ):
                    continue

                user_prompt = row["user_prompt"]

                criteria_verbalized = []
                for criteria in self.eval_config.config.rubric:
                    criteria_verbalized.append(f"{criteria.name}: {criteria.statement}")

                likert_scale_verbalized = LIKERT_PREBUILT_MAP[
                    self.eval_config.config.prebuilt_likert_scale
                ]

                # Get the judge prompt.
                judge_prompt = self.eval_config.config.prompt_template.format(
                    criteria_verbalized=criteria_verbalized,
                    likert_scale_verbalized=likert_scale_verbalized,
                    user_prompt=user_prompt,
                    response=row["completion_text"],
                )

                schema_class = create_dynamic_schema(self.eval_config)

                judging_tasks.append(
                    self.get_judge_rubric_structured_output(
                        user_prompt=user_prompt,
                        judge_prompt=judge_prompt,
                        judge_model=judge_model,
                        model_being_judged=model_being_judged,
                        schema_class=schema_class,
                    )
                )
        return judging_tasks

    async def get_judge_rubric_ratings(
        self,
        completions_df: pd.DataFrame,
        temperature: float | None = None,
    ) -> pd.DataFrame:
        """Get an async structured output task for the given prompt."""
        judging_tasks = await self.get_judge_rubric_tasks(
            completions_df, temperature=temperature
        )

        print(f"Generated {len(judging_tasks)} judging tasks.")

        judgments = []
        for future in tqdm.asyncio.tqdm.as_completed(
            judging_tasks, total=len(judging_tasks)
        ):
            result = await future

            # Replace 'structured_output' with its attributes as columns
            structured_output = result.pop("structured_output")
            result.update(structured_output.model_dump())
            judgments.append(result)

        # Add an overall score column, which is the mean of all criteria scores.
        for judgment in judgments:
            criteria_scores = [
                judgment[f"{criteria.name}"]
                for criteria in self.eval_config.config.rubric
            ]
            judgment["Overall"] = sum(criteria_scores) / len(criteria_scores)
        return pd.DataFrame(judgments)

    async def get_judge_pairwise_structured_output(
        self,
        user_prompt: str,
        completions_map: dict[str, str],
        llm1: str,
        llm2: str,
        judge_model: str,
        temperature: float | None = None,
    ):
        prompt_template = self.eval_config.config.prompt_template
        prompt_template_fields = {
            "user_prompt": user_prompt,
            "response_1": completions_map[llm1],
            "response_2": completions_map[llm2],
            "pairwise_comparison_labels": PAIRWISE_COMPARISON_LABEL_MAP[
                self.eval_config.config.granularity
            ],
        }

        schema_class = get_pairwise_comparison_schema(
            self.eval_config.config.granularity,
            self.eval_config.cot_enabled,
        )

        check_prompt_template_contains_all_placeholders(
            prompt_template, prompt_template_fields
        )

        judge_prompt = prompt_template.format(**prompt_template_fields)

        structured_output, completion = await self._with_rate_limit(
            # self.client.beta.chat.completions.parse(
            self.client_structured.chat.completions.create_with_completion(
                model=judge_model,
                messages=[
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=temperature,
                response_model=schema_class,
                extra_body={"provider": {"require_parameters": True}},
            )
        )

        return {
            "user_prompt": user_prompt,
            "judge_model": judge_model,
            "first_completion_by": llm1,
            "second_completion_by": llm2,
            "structured_output": structured_output,
            "temperature": temperature,
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    async def get_judge_pairwise_rating_tasks_for_single_prompt(
        self,
        user_prompt: str,
        completions_df: pd.DataFrame,
        temperature: float | None = None,
    ) -> list:
        completions_map = {
            row["model"]: row["completion_text"] for _, row in completions_df.iterrows()
        }

        pairwise_comparison_config = self.eval_config.config

        # Generate all pairs of completions.
        completion_pairs = [
            (llm1, llm2)
            for i, llm1 in enumerate(completions_map.keys())
            for llm2 in list(completions_map.keys())[i + 1 :]
        ]

        # Skip equal pairs.
        if pairwise_comparison_config.skip_equal_pairs:
            completion_pairs = [
                (llm1, llm2)
                for llm1, llm2 in completion_pairs
                if completions_map[llm1] != completions_map[llm2]
            ]

        # Apply positional flipping.
        if pairwise_comparison_config.position_flipping:
            completion_pairs = [
                (llm2, llm1) for llm1, llm2 in completion_pairs
            ] + completion_pairs

        # Filter down the pairs based on the pairwise_comparison_config.
        if pairwise_comparison_config.algorithm_type == "all_pairs":
            # No filtering needed.
            pass
        elif pairwise_comparison_config.algorithm_type == "random_pairs":
            # Generate a random sample of pairs of completions.
            completion_pairs = random.sample(
                completion_pairs,
                pairwise_comparison_config.n_random_pairs,
            )
        elif pairwise_comparison_config.algorithm_type == "fixed_reference_models":
            # Use llm1 as a fixed reference model.
            completion_pairs = [
                pair
                for pair in completion_pairs
                if pair[0]
                in pairwise_comparison_config.algorithm_config.reference_models
            ]

        # Apply reps.
        completion_pairs = [
            (llm1, llm2)
            for llm1, llm2 in completion_pairs
            for _ in range(self.eval_config.reps)
        ]

        # Fail if completion_pairs is empty.
        if len(completion_pairs) == 0:
            raise ValueError(
                "No pairs of completions to judge. Please check your pairwise_comparison_config. Perhaps reference models are misspelled?"
            )

        # Convert the completion_pairs into tasks.
        tasks = []
        for llm1, llm2 in completion_pairs:
            for judge_model in self.judge_models:
                if self.eval_config.exclude_self_grading and (
                    llm1 == judge_model or llm2 == judge_model
                ):
                    # Self-grading is disabled.
                    continue

                # Generate a judging task.
                tasks.append(
                    self.get_judge_pairwise_structured_output(
                        user_prompt=user_prompt,
                        completions_map=completions_map,
                        llm1=llm1,
                        llm2=llm2,
                        judge_model=judge_model,
                        temperature=temperature,
                    )
                )

        return tasks

    async def get_judge_pairwise_rating_tasks(
        self,
        completions_df: pd.DataFrame,
        temperature: float | None = None,
    ) -> list:
        # Create a map of user_prompt -> model -> completion.
        user_prompts = completions_df["user_prompt"].unique()

        tasks = []
        for user_prompt in user_prompts:
            tasks.extend(
                await self.get_judge_pairwise_rating_tasks_for_single_prompt(
                    user_prompt=user_prompt,
                    completions_df=completions_df[
                        completions_df["user_prompt"] == user_prompt
                    ],
                    temperature=temperature,
                )
            )
        return tasks

    async def get_judge_pairwise_ratings(
        self,
        completions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        temperature = self.eval_config.temperature
        judging_tasks = await self.get_judge_pairwise_rating_tasks(
            completions_df,
        )

        print(f"Generated {len(judging_tasks)} judging tasks.")

        judgments = []
        for future in tqdm.asyncio.tqdm.as_completed(
            judging_tasks, total=len(judging_tasks)
        ):
            result = await future

            # Replace 'structured_output' with its attributes as columns
            structured_output = result.pop("structured_output")
            result.update(structured_output.model_dump())

            # Process the pairwise choice to remove brackets.
            result["pairwise_choice"] = process_pairwise_choice(
                result["pairwise_choice"]
            )
            judgments.append(result)

        return judgments

    async def judge(
        self,
        completions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.eval_config.type == "direct_assessment":
            judgments = await self.get_judge_rubric_ratings(completions_df)
        elif self.eval_config.type == "pairwise_comparison":
            judgments = await self.get_judge_pairwise_ratings(completions_df)
        else:
            raise ValueError(
                f"Invalid evaluation config type: {self.eval_config.type}. Must be one of: direct_assessment, pairwise_comparison."
            )
        return pd.DataFrame(judgments)

    def get_judging_df(self) -> pd.DataFrame:
        """Returns the judgments made by the council."""
        return pd.DataFrame(self.judgments)

    def get_completions_df(self) -> pd.DataFrame:
        """Returns the completions made by the council."""
        return pd.DataFrame(self.completions)

    async def execute(self, prompts: str | list[str]):
        # Normalize to list[str]
        if isinstance(prompts, str):
            prompts = [prompts]

        # ─── 1️⃣  completions phase ─────────
        completion_tasks = [
            task
            for p in prompts
            for task in await self.get_text_completions(p, self.eval_config.temperature)
        ]

        print(f"Generated {len(completion_tasks)} completion tasks.")

        completions_raw = [
            await fut
            for fut in tqdm.asyncio.tqdm.as_completed(
                completion_tasks, total=len(completion_tasks)
            )
        ]
        completions_df = pd.DataFrame(completions_raw)
        self.completions.extend(completions_raw)
        self.user_prompts.extend(prompts)

        # ─── 2️⃣  judging phase ─────────
        judging_df = await self.judge(
            completions_df=completions_df,
        )
        self.judgments.extend(judging_df.to_dict("records"))
        return completions_df, judging_df

    def leaderboard(self, outfile: str | None = None) -> pd.DataFrame:
        if self.eval_config.type == "pairwise_comparison":
            judging_df = self.get_judging_df()
            reference_llm = get_reference_llm(
                judging_df,
                self.eval_config,
            )

            rankings_results = analyze_rankings_separability_polarization(
                judging_df,
                reference_llm_respondent=reference_llm,
                bootstrap_rounds=10,
                include_individual_judges=True,
                include_council_majority=True,
                include_council_mean_pooling=True,
                include_council_no_aggregation=True,
                example_id_column="user_prompt",
            )

            if outfile:
                show = False
            else:
                show = True
            plot_arena_hard_elo_stats(
                rankings_results["council/no-aggregation"]["elo_scores"],
                "",
                outfile=outfile,
                show=show,
            )
            return rankings_results
        elif self.eval_config.type == "direct_assessment":
            return plot_direct_assessment_charts(
                self.get_judging_df(), self.eval_config, outfile=outfile
            )
        else:
            raise ValueError(
                f"Unimplemented leaderboard for evaluation config type: {self.eval_config.type}. Must be one of: direct_assessment, pairwise_comparison."
            )

    def win_rate_heatmap(self) -> pd.DataFrame:
        if self.eval_config.type != "pairwise_comparison":
            raise ValueError(
                "Win rate heatmap can only be generated for pairwise comparison evaluations."
            )
        return bradley_terry_analysis(self.get_judging_df())

    def explicit_win_rate_heatmap(self):
        if self.eval_config.type != "pairwise_comparison":
            raise ValueError(
                "Explicit win rate heatmap can only be generated for pairwise comparison evaluations."
            )

        judging_df = self.get_judging_df()
        win_rate_map = get_explicit_win_rates(judging_df)
        num_models = len(self.models)
        figsize = (max(5, num_models), max(4, num_models))

        plot_heatmap(
            win_rate_map,
            ylabel="Respondent",
            xlabel="vs. Respondent",
            vmin=0,
            vmax=1,
            cmap="coolwarm",
            outfile=None,
            figsize=figsize,
            font_size=8,
        )

        return win_rate_map

    def judge_agreement(self, show_plots=True):
        if self.eval_config.type == "direct_assessment":
            judging_df = self.get_judging_df()

            agreement_matrices, mean_agreement_df = get_judge_agreement(
                judging_df, self.eval_config
            )

            if show_plots:
                # Plot agreement matrices for each criterion
                for crit_col, matrix in agreement_matrices.items():
                    plot_heatmap(
                        matrix,
                        ylabel="Judge Model",
                        xlabel="Judge Model",
                        title=f"Judge Agreement Matrix: {crit_col}",
                        vmin=0,
                        vmax=1,
                        cmap="coolwarm_r",
                        outfile=None,
                        figsize=(8, 6),
                        font_size=8,
                    )

            return agreement_matrices, mean_agreement_df

        elif self.eval_config.type == "pairwise_comparison":
            judge_agreement_map, mean_agreement_df = get_judge_agreement_map(
                self.get_judging_df(), example_id_column="user_prompt"
            )
            if show_plots:
                # Plot the judge agreement map as a heatmap
                for judge_model, agreement_matrix in judge_agreement_map.items():
                    plot_heatmap(
                        agreement_matrix,
                        ylabel="Judge Model",
                        xlabel="Judge Model",
                        title=f"Pairwise Judge Agreement: {judge_model}",
                        vmin=0,
                        vmax=1,
                        cmap="coolwarm",
                        outfile=None,
                        figsize=(8, 6),
                        font_size=8,
                    )
            return judge_agreement_map, mean_agreement_df

    def affinity(self, show_plots=True):
        if self.eval_config.type == "direct_assessment":
            affinity_matrices = get_affinity_matrices(
                self.get_judging_df(), self.eval_config
            )
            judge_models = self.judge_models
            models_being_judged = self.models

            if show_plots:
                for crit, matrix in affinity_matrices.items():
                    # Plot heatmap
                    plot_heatmap(
                        matrix,
                        ylabel="Model Being Judged",
                        xlabel="Judge Model",
                        title=f"Affinity Heatmap: {crit}",
                        vmin=0,
                        vmax=1,
                        cmap="coolwarm",
                        outfile=None,
                        figsize=(
                            max(8, len(judge_models) + 2),
                            max(6, len(models_being_judged)),
                        ),
                        font_size=8,
                    )

            return affinity_matrices
        elif self.eval_config.type == "pairwise_comparison":
            judging_df = self.get_judging_df()
            reference_llm = get_reference_llm(
                judging_df,
                self.eval_config,
            )
            affinity_results = get_affinity_df(
                judging_df,
                reference_llm_respondent=reference_llm,
                example_id_column="user_prompt",
            )
            if show_plots:
                if "judge_preferences" in affinity_results:
                    plot_heatmap(
                        affinity_results["judge_preferences"],
                        ylabel="Judge Model",
                        xlabel="Model Being Judged",
                        title="Judge Preferences",
                        vmin=0,
                        vmax=1,
                        cmap="coolwarm",
                        outfile=None,
                        figsize=(
                            max(8, len(self.judge_models) + 2),
                            max(6, len(self.models)),
                        ),
                        font_size=8,
                    )
                if "judge_preferences_council_normalized" in affinity_results:
                    plot_heatmap(
                        affinity_results["judge_preferences_council_normalized"],
                        ylabel="Judge Model",
                        xlabel="Model Being Judged",
                        title="Judge Preferences (Council Normalized)",
                        vmin=-1,
                        vmax=1,
                        cmap="coolwarm",
                        outfile=None,
                        figsize=(
                            max(8, len(self.judge_models) + 2),
                            max(6, len(self.models)),
                        ),
                        font_size=8,
                    )
            return affinity_results

    def generate_hf_readme(self) -> str:
        """
        Generate a markdown string describing the dataset for Hugging Face Hub.
        """

        def make_serializable(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            elif hasattr(obj, "__dict__"):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return obj

        eval_config_str = json.dumps(
            make_serializable(self.eval_config),
            indent=2,
        )
        num_prompts = len(set(self.user_prompts))
        models_judged = "\n- " + "\n- ".join(self.models)
        judge_models = (
            "\n- " + "\n- ".join(self.judge_models)
            if hasattr(self, "judge_models") and self.judge_models
            else models_judged
        )

        readme = f"""
## Leaderboard

![Leaderboard](leaderboard.png)

## Dataset Overview

**Number of unique prompts:** {num_prompts}

**Models evaluated:** {models_judged}

**Judge models:** {judge_models}

**Provider:** [OpenRouter](https://openrouter.ai)

## Evaluation Configuration
```json
{eval_config_str}
```

## About

This dataset was generated using the [LLM Council](https://github.com/llm-council/lm-council), a framework for evaluating language models by having them judge each other democratically.
"""

        return readme

    def upload_to_hf(self, repo_id: str):
        """Upload completions, judgments, leaderboard figure, and README to Hugging Face Hub as a dataset."""

        # Prepare datasets
        completions_ds = Dataset.from_pandas(
            pd.DataFrame(self.completions), preserve_index=False
        )
        judgments_ds = Dataset.from_pandas(
            pd.DataFrame(self.judgments), preserve_index=False
        )

        # Generate and save leaderboard figure
        leaderboard_df = self.leaderboard("leaderboard.png")

        leaderboard_ds = Dataset.from_pandas(
            leaderboard_df,
            preserve_index=False,
        )

        # Push each dataset to Hugging Face Hub with a config_name and split
        completions_ds.push_to_hub(repo_id, config_name="completions")
        judgments_ds.push_to_hub(repo_id, config_name="judgments")
        leaderboard_ds.push_to_hub(repo_id, config_name="leaderboard")

        # Upload leaderboard.png to the repo
        api = HfApi()
        api.upload_file(
            path_or_fileobj="leaderboard.png",
            path_in_repo="leaderboard.png",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add leaderboard figure",
        )
        os.remove("leaderboard.png")

        # Push README to the Hugging Face Hub
        readme_str = self.generate_hf_readme()
        # Get the current README if it exists
        try:
            old_readme = api.hf_hub_download(repo_id, "README.md", repo_type="dataset")
            with open(old_readme, "r", encoding="utf-8") as f:
                current_readme = f.read()
        except Exception:
            current_readme = ""

        # Append the new readme content
        combined_readme = current_readme + "\n\n" + readme_str

        api.upload_file(
            path_or_fileobj=combined_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Append to README.md",
        )

    def save(self, outdir):
        # Save all artifacts to a directory.
        """
        outdir/
            models.json
            completions.jsonl
            judge_ratings.jsonl
            user_prompts.jsonl
            eval_config.json
        """
        os.makedirs(outdir, exist_ok=True)

        with open(os.path.join(outdir, "models.json"), "w") as f:
            json.dump(self.models, f)
        with open(os.path.join(outdir, "user_prompts.json"), "w") as f:
            json.dump(self.user_prompts, f)

        pd.DataFrame(self.completions).to_json(
            os.path.join(outdir, "completions.jsonl"), orient="records", lines=True
        )
        pd.DataFrame(self.judgments).to_json(
            os.path.join(outdir, "judge_ratings.jsonl"), orient="records", lines=True
        )
        self.eval_config.save_config(os.path.join(outdir, "eval_config.json"))

    @staticmethod
    def load(indir: str) -> "LanguageModelCouncil":
        """
        Load a LanguageModelCouncil instance from a directory.

        indir/
            models.json
            completions.jsonl
            judge_ratings.jsonl
            user_prompts.jsonl
            eval_config.json
        """
        # If indir is not an absolute path, make it relative to the current working directory
        if not os.path.isabs(indir):
            indir = os.path.abspath(os.path.join(os.getcwd(), indir))

        # Load LLMs
        with open(os.path.join(indir, "models.json"), "r") as f:
            models = json.load(f)

        # Load prompts
        with open(os.path.join(indir, "user_prompts.json"), "r") as f:
            user_prompts = json.load(f)

        # Load completions
        completions = pd.read_json(
            os.path.join(indir, "completions.jsonl"), orient="records", lines=True
        ).to_dict(orient="records")

        # Load judgments
        judgments = pd.read_json(
            os.path.join(indir, "judge_ratings.jsonl"), orient="records", lines=True
        ).to_dict(orient="records")

        # Load evaluation config
        eval_config = EvaluationConfig.load_config(
            os.path.join(indir, "eval_config.json")
        )

        # Create the instance
        council = LanguageModelCouncil(models=models, eval_config=eval_config)
        council.user_prompts = user_prompts
        council.completions = completions
        council.judgments = judgments

        return council
