from llm_council.providers.together_provider import TogetherProvider
import asyncio
from llm_council.structured_outputs import STRUCTURED_OUTPUT_REGISTRY
from tqdm.asyncio import tqdm


def test_async_client():
    """Sample code that tests openai async client."""
    provider = TogetherProvider(
        "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    )

    async def get_all_tasks() -> tuple[list, list]:
        tasks = [provider.get_async_completion_task("Say Hello.") for _ in range(3)]
        extractions = []
        completions = []
        for result in tqdm.as_completed(tasks, total=len(tasks)):
            extraction, completion = await result
            extractions.append(extraction)
            completions.append(completion)
        return extractions, completions

    extractions, completions = asyncio.run(get_all_tasks())

    assert len(extractions) == 3
    assert len(completions) == 3


def test_async_client_structured_output():
    """Sample code that tests openai async client with structured outputs."""
    provider = TogetherProvider(
        "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    )

    async def get_all_tasks() -> tuple[list, list]:
        tasks = [
            provider.get_async_completion_task(
                "Create a hypothetical user.", schema_name="test_user"
            )
            for _ in range(3)
        ]

        extractions = []
        completions = []
        for result in tqdm.as_completed(tasks, total=len(tasks)):
            extraction, completion = await result
            extractions.append(extraction)
            completions.append(completion)
        return extractions, completions

    extractions, completions = asyncio.run(get_all_tasks())

    assert len(extractions) == 3
    assert isinstance(extractions[0], STRUCTURED_OUTPUT_REGISTRY["test_user"])
    assert len(completions) == 3
