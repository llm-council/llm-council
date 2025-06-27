import pandas as pd
import requests
import os
import json
from llm_council.judging.schema import (
    EvaluationConfig,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_PAIRWISE_EVALUATION_CONFIG,
)
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
from llm_council.analysis.visualization import plot_arena_hard_elo_stats
from llm_council.analysis.pairwise.separability import (
    analyze_rankings_separability_polarization,
)
import aiohttp
from aiolimiter import AsyncLimiter
import instructor
from llm_council.analysis.pairwise.explicit_win_rate import get_explicit_win_rates
from llm_council.analysis.pairwise.agreement import get_judge_agreement_map


def process_pairwise_choice(raw_pairwise_choice: str) -> str:
    return raw_pairwise_choice.replace("[", "").replace("]", "")


class LanguageModelCouncil:

    def __init__(
        self,
        models: list[str],
        judge_models: list[str] | None = None,
        eval_config: EvaluationConfig | None = None,
    ):
        self.models = models

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

        # Define the evaluation config if one is not already defined.
        if eval_config is None:
            self.eval_config = DEFAULT_PAIRWISE_EVALUATION_CONFIG
            # self.eval_config = DEFAULT_EVALUATION_CONFIG
        else:
            self.eval_config = eval_config

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

        # Execute all tasks concurrently.
        judgments = []
        for future in tqdm.asyncio.tqdm.as_completed(
            judging_tasks, total=len(judging_tasks)
        ):
            result = await future

            # Replace 'structured_output' with its attributes as columns
            structured_output = result.pop("structured_output")
            result.update(structured_output.model_dump())
            judgments.append(result)

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

        # structured_output = completion.choices[0].message.parsed

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

    def leaderboard(self):
        if self.eval_config.type == "pairwise_comparison":
            judging_df = self.get_judging_df()
            reference_llm_respondent = judging_df.iloc[0]["first_completion_by"]
            rankings_results = analyze_rankings_separability_polarization(
                judging_df,
                reference_llm_respondent=reference_llm_respondent,
                bootstrap_rounds=10,
                include_individual_judges=True,
                include_council_majority=True,
                include_council_mean_pooling=True,
                include_council_no_aggregation=True,
                example_id_column="user_prompt",
            )

            plot_arena_hard_elo_stats(
                rankings_results["council/no-aggregation"]["elo_scores"],
                "",
                None,
                show=True,
            )
            return rankings_results
        else:
            raise ValueError(
                "Leaderboard can only be generated for pairwise comparison evaluations."
            )

    def win_rate_heatmap(self):
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

    def judge_agreement(self):
        return get_judge_agreement_map(
            self.get_judging_df(), example_id_column="user_prompt"
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
