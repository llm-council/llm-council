from llm_council.providers.together_provider import TogetherProvider
import asyncio
import tqdm.asyncio
from llm_council.structured_outputs import STRUCTURED_OUTPUT_REGISTRY


def test_async_client():
    """Sample code that tests TogetherProvider async client."""
    provider = TogetherProvider(
        "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    )

    async def get_all_tasks() -> list:
        tasks = [
            provider.get_async_completion_task(
                "Say Hello.",
                task_metadata={
                    "llm": "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
                },
            )
            for _ in range(3)
        ]

        completion_texts = []
        for future in tqdm.asyncio.tqdm.as_completed(tasks, total=len(tasks)):
            result = await future
            task_metadata = result["task_metadata"]
            llm = task_metadata["llm"]
            completion_text = result["completion_text"]
            completion_texts.append(completion_text)
        return completion_texts

    completion_texts = asyncio.run(get_all_tasks())

    assert len(completion_texts) == 3


def test_async_client_structured_output():
    """Sample code that tests TogetherProvider async client with structured outputs."""
    provider = TogetherProvider(
        "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    )

    async def get_all_tasks() -> list:
        tasks = [
            provider.get_async_completion_task(
                "Create a hypothetical user.",
                schema_class_path="llm_council.structured_outputs.User",
                task_metadata={
                    "llm": "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
                },
            )
            for _ in range(3)
        ]

        structured_outputs = []
        for future in tqdm.asyncio.tqdm.as_completed(tasks, total=len(tasks)):
            result = await future
            task_metadata = result["task_metadata"]
            llm = task_metadata["llm"]
            structured_output = result["structured_output"]
            structured_outputs.append(structured_output)

        return structured_outputs

    structured_outputs = asyncio.run(get_all_tasks())

    assert len(structured_outputs) == 3
    # First item of the list, and the first item of the tuple is the actual structured output completion.
    assert structured_outputs[0].__class__.__name__ == "User"
