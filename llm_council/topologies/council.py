import pandas as pd
from llm_council.judging.schema import (
    EvaluationConfig,
    create_dynamic_schema,
    DEFAULT_EVALUATION_CONFIG,
)
from llm_council.topologies.base_topology import topology
from llm_council.providers.base_provider import (
    get_provider_instance_for_llm,
    PROVIDER_REGISTRY,
)
import tqdm.asyncio
import asyncio
from llm_council.providers.utils import (
    check_prompt_template_contains_all_placeholders,
)
from llm_council.providers.base_provider import BaseProvider
from llm_council.judging.prompt_builder import LIKERT_PREBUILT_MAP
from llm_council.sessions.council_session import CouncilSession


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

    # sample_return_object = hypothesis_jsonschema.from_schema(
    #     schema_class.schema()
    # ).example()

    # breakpoint()

    prompt = prompt_template.format(
        criteria_verbalized=criteria_verbalized,
        likert_scale_verbalized=likert_scale_verbalized,
        # sample_return_object=sample_return_object,
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


async def get_async_judging_task(
    provider_instance: BaseProvider,
    eval_config: EvaluationConfig,
    prompt_template_fields=dict,
    task_metadata: dict = dict,
    temperature: float | None = None,
):
    if eval_config.type == "direct_assessment":
        return await get_async_judge_direct_assessment_task(
            provider_instance,
            eval_config,
            prompt_template_fields=prompt_template_fields,
            task_metadata=task_metadata,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported evaluation type: {eval_config.type}")


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

    def __init__(self, llms: list[str], allowed_providers: list[str] | None = None):
        """allowed_providers is a list of provider names that are allowed to be used. If None, then all providers are allowed."""
        # Hold it all in memory bro.
        # LLMs can be specified by short name or full name.

        # Create a map of llm -> initialized provider object for each llm
        self.llm_to_provider_map = {
            llm: get_provider_instance_for_llm(llm, allowed_providers) for llm in llms
        }

    async def collect_completions(
        self, prompt: str
    ) -> tuple[dict[str, str], dict[str, dict]]:
        # Create a list of completion tasks.
        completion_tasks = [
            provider.get_async_completion_task(prompt, task_metadata={"llm": llm})
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
                    "llm": llm,
                    "completion_text": completion_text,
                    **get_usage_information(completion),
                }
            )

        return pd.DataFrame(completions)

    async def collect_judge_ratings(
        self,
        user_prompt: str,
        completions_df: pd.DataFrame,
        evaluation_config: EvaluationConfig | None,
    ) -> tuple[pd.DataFrame, EvaluationConfig]:
        # Create a map of llm -> completion for that LLM.
        completions_map = {
            row["llm"]: row["completion_text"] for _, row in completions_df.iterrows()
        }

        # Define the evaluation config if one is not already defined.
        if evaluation_config is None:
            evaluation_config = DEFAULT_EVALUATION_CONFIG

        # TODO: Support pairwise judging.
        if evaluation_config.type == "direct_assessment":
            # Direct assessment.
            judging_tasks = []

            # Go through all completions and all judges and generate completion tasks.
            for llm_responder, completion_text in completions_map.items():
                for llm_judge, provider in self.llm_to_provider_map.items():
                    if (
                        evaluation_config.exclude_self_grading
                        and llm_responder == llm_judge
                    ):
                        # Self-grading is disabled.
                        continue

                    # Generate a judging task.
                    judging_tasks.append(
                        get_async_judging_task(
                            provider_instance=provider,
                            eval_config=evaluation_config,
                            prompt_template_fields={
                                "user_prompt": user_prompt,
                                "response": completion_text,
                            },
                            task_metadata={
                                "llm_responder": llm_responder,
                                "llm_judge": llm_judge,
                            },
                        )
                    )

            # Collect judgments.
            judgments = []
            for future in tqdm.asyncio.tqdm.as_completed(
                judging_tasks, total=len(judging_tasks)
            ):
                result = await future
                task_metadata = result["task_metadata"]
                llm_responder = task_metadata["llm_responder"]
                llm_judge = task_metadata["llm_judge"]

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
                        **criteria_map,
                        **get_usage_information(judgment_completion),
                    }
                )

            return pd.DataFrame(judgments), evaluation_config

        # For pairwise judging, we need to generate pairs of completions first.

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

        # Judging.
        judging_df, evaluation_config = asyncio.run(
            self.collect_judge_ratings(
                user_prompt=prompt,
                completions_df=completions_df,
                evaluation_config=evaluation_config,
            )
        )

        # TODO: Add judging llms.
        session = CouncilSession(
            llms=list(self.llm_to_provider_map.keys()),
            prompt=prompt,
            completions_df=completions_df,
            judging_df=judging_df,
            evaluation_config=evaluation_config,
        )

        return session


# The council executes many sessions.
# Should there be a way to combine sessions.

# Users may also be interested in running the same completions through different judging configurations.
# Future expansions / use cases:
# Support automatic evaluation?
# Support consistency experiments?


if __name__ == "__main__":
    # Quick test.
    lmc = LanguageModelCouncil(
        # TODO: Enable this concise specification.
        # llms=["gpt-4o-mini", "Llama-3.1-8B"]
        llms=[
            "openai://gpt-4o-mini",
            "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            # "anthropic://claude-3-haiku-20240307",
        ]
    )

    session = lmc.execute(prompt="Say hello.")

    session.save("tests/testdata/sample_session")
