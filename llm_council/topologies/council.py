import pandas as pd
import os
import json
from llm_council.judging.schema import (
    EvaluationConfig,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_PAIRWISE_EVALUATION_CONFIG,
)
from llm_council.structured_outputs import create_dynamic_schema
from llm_council.topologies.base_topology import topology
from llm_council.providers.base_provider import (
    get_provider_instance_for_llm,
    PROVIDER_REGISTRY,
)
import tqdm.asyncio
import random
import asyncio
from llm_council.providers.utils import (
    check_prompt_template_contains_all_placeholders,
)
from llm_council.providers.base_provider import (
    BaseProvider,
    get_allowed_providers_from_env,
)
from llm_council.judging.prompt_builder import LIKERT_PREBUILT_MAP
from llm_council.sessions.council_session import CouncilSession
from llm_council.structured_outputs import (
    PAIRWISE_COMPARISON_LABEL_MAP,
    get_pairwise_comparison_schema,
)


async def get_async_judge_pairwise_comparison_task(
    provider_instance: BaseProvider,
    eval_config: EvaluationConfig,
    prompt_template_fields=dict,
    task_metadata: dict = dict,
    temperature: float | None = None,
):
    prompt_template = eval_config.config.prompt_template

    # Get pairwise comparison labels.
    prompt_template_fields["pairwise_comparison_labels"] = (
        PAIRWISE_COMPARISON_LABEL_MAP[eval_config.config.granularity]
    )

    schema_class = get_pairwise_comparison_schema(
        eval_config.config.granularity, eval_config.cot_enabled
    )

    check_prompt_template_contains_all_placeholders(
        prompt_template, prompt_template_fields
    )

    prompt = prompt_template.format(**prompt_template_fields)

    return await provider_instance.get_async_completion_task(
        prompt, task_metadata, temperature=temperature, schema_class=schema_class
    )


async def get_async_judge_direct_assessment_task(
    provider_instance: BaseProvider,
    eval_config: EvaluationConfig,
    prompt_template_fields=dict,
    task_metadata: dict = dict,
    temperature: float | None = None,
):
    prompt_template = eval_config.config.prompt_template
    schema_class = create_dynamic_schema(eval_config)

    check_prompt_template_contains_all_placeholders(
        prompt_template, prompt_template_fields
    )

    # Add the criteria.
    criteria_verbalized = []
    for criteria in eval_config.config.criteria:
        criteria_verbalized.append(f"{criteria.name}: {criteria.criteria_statement}")

    likert_scale_verbalized = LIKERT_PREBUILT_MAP[
        eval_config.config.prebuilt_likert_scale
    ]

    prompt = prompt_template.format(
        criteria_verbalized=criteria_verbalized,
        likert_scale_verbalized=likert_scale_verbalized,
        **prompt_template_fields,
    )

    prompt += """Please return your rating as a JSON object with the key as the criteria and the rating as an integer value. For example:

{
    <criteria_name>: <rating>,
}
"""

    return await provider_instance.get_async_completion_task(
        prompt, task_metadata, temperature=temperature, schema_class=schema_class
    )


def get_usage_information(completion):
    """Returns usage information from the different completion objects."""
    if completion.__class__.__module__ == "anthropic.types.message":
        return {
            "prompt_tokens": completion.usage.input_tokens,
            "completion_tokens": completion.usage.output_tokens,
            "total_tokens": completion.usage.input_tokens
            + completion.usage.output_tokens,
        }
    else:
        return {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }


@topology(topology_name="council")
class LanguageModelCouncil:

    def __init__(
        self, llms: list[str], evaluation_config: EvaluationConfig | None = None
    ):
        allowed_providers = get_allowed_providers_from_env()

        # Create a map of llm -> initialized provider object for each llm
        self.llm_to_provider_map = {
            llm: get_provider_instance_for_llm(llm, allowed_providers) for llm in llms
        }

        # Define the evaluation config if one is not already defined.
        if evaluation_config is None:
            self.evaluation_config = DEFAULT_PAIRWISE_EVALUATION_CONFIG
        else:
            self.evaluation_config = evaluation_config

        self.user_prompts = []

        # List of all completions.
        self.completions = []

        # List of all judgments.
        self.judgments = []

    async def collect_completions(
        self, user_prompt: str
    ) -> tuple[dict[str, str], dict[str, dict]]:
        # Create a list of completion tasks.
        completion_tasks = [
            provider.get_async_completion_task(user_prompt, task_metadata={"llm": llm})
            for llm, provider in self.llm_to_provider_map.items()
        ]

        # Collect completions.
        completions = []
        for future in tqdm.asyncio.tqdm.as_completed(
            completion_tasks, total=len(completion_tasks)
        ):
            result = await future

            task_metadata = result["task_metadata"]
            llm = task_metadata["llm"]
            completion_text = result["completion_text"]
            completion = result["completion"]

            completions.append(
                {
                    "user_prompt": user_prompt,
                    "llm": llm,
                    "completion_text": completion_text,
                    **get_usage_information(completion),
                }
            )

        self.user_prompts.append(user_prompt)
        return pd.DataFrame(completions)

    async def collect_judge_ratings(
        self,
        user_prompt: str,
        completions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # Create a map of llm -> completion.
        completions_map = {
            row["llm"]: row["completion_text"] for _, row in completions_df.iterrows()
        }

        if self.evaluation_config.type == "direct_assessment":
            judging_tasks = self.get_direct_assessment_judging_tasks(
                completions_map, user_prompt
            )
            judgments = []
            for future in tqdm.asyncio.tqdm.as_completed(
                judging_tasks, total=len(judging_tasks)
            ):
                result = await future

                task_metadata = result["task_metadata"]
                llm_responder = task_metadata["llm_responder"]
                llm_judge = task_metadata["llm_judge"]
                user_prompt = task_metadata["prompt"]

                # Extract the criteria map.
                structured_output = result["structured_output"]
                criteria_map = {
                    criteria.name: getattr(structured_output, criteria.name)
                    for criteria in self.evaluation_config.config.criteria
                }

                judgment_completion = result["completion"]

                judgments.append(
                    {
                        "llm_responder": llm_responder,
                        "llm_judge": llm_judge,
                        "user_prompt": user_prompt,
                        **criteria_map,
                        **get_usage_information(judgment_completion),
                    }
                )
            return pd.DataFrame(judgments)

        elif self.evaluation_config.type == "pairwise_comparison":
            judging_tasks = self.get_pairwise_comparison_judging_tasks(
                completions_map, user_prompt
            )
            judgments = []
            for future in tqdm.asyncio.tqdm.as_completed(
                judging_tasks, total=len(judging_tasks)
            ):
                result = await future

                # Extract relevant results.
                task_metadata = result["task_metadata"]
                llm_responder_1 = task_metadata["llm_responder_1"]
                llm_responder_2 = task_metadata["llm_responder_2"]
                llm_judge = task_metadata["llm_judge"]
                user_prompt = task_metadata["user_prompt"]
                structured_output = result["structured_output"]
                judgment_completion = result["completion"]

                judgment_row = {
                    "user_prompt": user_prompt,
                    "llm_judge": llm_judge,
                    "llm_responder_1": llm_responder_1,
                    "llm_responder_2": llm_responder_2,
                    "rating": structured_output.rating,
                }

                # Add explanation if it's available.
                if hasattr(structured_output, "explanation"):
                    judgment_row["explanation"] = structured_output.explanation

                # Add usage information.
                judgment_row.update(
                    {
                        **get_usage_information(judgment_completion),
                    }
                )

                judgments.append(judgment_row)
            return pd.DataFrame(judgments)

    def get_pairwise_comparison_judging_tasks(self, completions_map, user_prompt):
        pairwise_comparison_config = self.evaluation_config.config

        # Generate all pairs of completions.
        completion_pairs = [
            (llm1, llm2)
            for i, llm1 in enumerate(completions_map.keys())
            for llm2 in list(completions_map.keys())[i + 1 :]
        ]

        # Filter down the pairs based on the pairwise_comparison_config.
        if pairwise_comparison_config.algorithm_type == "all_pairs":
            # no filtering needed.
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
                if pair[0] in pairwise_comparison_config.reference_models
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

        # Apply reps.
        completion_pairs = [
            (llm1, llm2)
            for llm1, llm2 in completion_pairs
            for _ in range(self.evaluation_config.reps)
        ]

        # Fail if completion_pairs is empty.
        if len(completion_pairs) == 0:
            raise ValueError(
                "No pairs of completions to judge. Please check your pairwise_comparison_config. Perhaps reference models are misspelled?"
            )

        # Convert the completion_pairs into tasks.
        pairwise_comparison_tasks = []
        for llm1, llm2 in completion_pairs:
            # Generate a judging task, for each LLM in the llm_to_provider_map.
            for llm_judge, provider in self.llm_to_provider_map.items():
                if self.evaluation_config.exclude_self_grading and (
                    llm1 == llm_judge or llm2 == llm_judge
                ):
                    # Self-grading is disabled.
                    continue

                # Generate a judging task.
                pairwise_comparison_tasks.append(
                    get_async_judge_pairwise_comparison_task(
                        provider_instance=provider,
                        eval_config=self.evaluation_config,
                        prompt_template_fields={
                            "user_prompt": user_prompt,
                            "response_1": completions_map[llm1],
                            "response_2": completions_map[llm2],
                        },
                        task_metadata={
                            "llm_responder_1": llm1,
                            "llm_responder_2": llm2,
                            "llm_judge": llm_judge,
                            "user_prompt": user_prompt,
                        },
                    )
                )

        print(f"Generated {len(pairwise_comparison_tasks)} pairwise comparison tasks.")
        if len(pairwise_comparison_tasks) == 0:
            raise ValueError(
                "No pairwise comparison tasks generated. Please check your pairwise_comparison_config."
            )

        return pairwise_comparison_tasks

    def get_direct_assessment_judging_tasks(self, completions_map, user_prompt) -> list:
        # Go through all completions and all judges and generate completion tasks.
        judging_tasks = []
        for llm_responder, completion_text in completions_map.items():
            for llm_judge, provider in self.llm_to_provider_map.items():
                if (
                    self.evaluation_config.exclude_self_grading
                    and llm_responder == llm_judge
                ):
                    # Self-grading is disabled.
                    continue

                # Generate a judging task.
                judging_tasks.append(
                    get_async_judge_direct_assessment_task(
                        provider_instance=provider,
                        eval_config=self.evaluation_config,
                        prompt_template_fields={
                            "user_prompt": user_prompt,
                            "response": completion_text,
                        },
                        task_metadata={
                            "llm_responder": llm_responder,
                            "llm_judge": llm_judge,
                            "prompt": user_prompt,
                        },
                    )
                )
        return judging_tasks

    async def execute_direct_assessment_judgment_tasks(
        judging_tasks, evaluation_config
    ) -> pd.DataFrame:
        judgments = []
        for future in tqdm.asyncio.tqdm.as_completed(
            judging_tasks, total=len(judging_tasks)
        ):
            result = await future
            task_metadata = result["task_metadata"]
            llm_responder = task_metadata["llm_responder"]
            llm_judge = task_metadata["llm_judge"]
            user_prompt = task_metadata["prompt"]

            # Extract the criteria map.
            structured_output = result["structured_output"]
            criteria_map = {
                criteria.name: getattr(structured_output, criteria.name)
                for criteria in evaluation_config.config.criteria
            }

            judgment_completion = result["completion"]

            judgments.append(
                {
                    "llm_responder": llm_responder,
                    "llm_judge": llm_judge,
                    "user_prompt": user_prompt,
                    **criteria_map,
                    **get_usage_information(judgment_completion),
                }
            )

        return pd.DataFrame(judgments)

    def execute(
        self,
        prompt: str | None,
        completions_df: pd.DataFrame | None = None,
        evaluation_config: EvaluationConfig | None = None,
        judging_llms: list[str] | None = None,
    ) -> CouncilSession:
        """Executes the council on the given prompt.

        If judging_llms is specified, then only those LLMs will be used for judging.

        If completions is specified, then the council will use those completions for judging.
        """
        # Fail if prompt and completions are both not None.
        if prompt is None and completions_df is None:
            raise ValueError("Only one of `prompt` or `completions` must be specified.")

        completions_df = asyncio.run(self.collect_completions(prompt))
        # add completions to self.completions.
        for _, row in completions_df.iterrows():
            self.completions.append(row)

        # Judging.
        judging_df = asyncio.run(
            self.collect_judge_ratings(
                user_prompt=prompt,
                completions_df=completions_df,
            )
        )
        # add judging_df to overall council self.judgments.
        for _, row in judging_df.iterrows():
            self.judgments.append(row)

    def save(self, outdir):
        # Save all artifacts to a directory.
        """
        outdir/
            llm_to_provider_map.json
            completions.jsonl
            judging.jsonl
            prompts.jsonl
            evaluation_config.json
        """
        os.makedirs(outdir, exist_ok=True)

        llms = list(self.llm_to_provider_map.keys())

        with open(os.path.join(outdir, "llms.json"), "w") as f:
            json.dump(llms, f)
        with open(os.path.join(outdir, "prompts.json"), "w") as f:
            json.dump(self.user_prompts, f)

        pd.DataFrame(self.completions).to_json(
            os.path.join(outdir, "completions.jsonl"), orient="records", lines=True
        )
        pd.DataFrame(self.judgments).to_json(
            os.path.join(outdir, "judging.jsonl"), orient="records", lines=True
        )
        self.evaluation_config.save_config(
            os.path.join(outdir, "evaluation_config.json")
        )
