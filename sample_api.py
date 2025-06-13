# """
# pip install llm_council
# """

# from llm_council import Council

# # Print a table of viable LLMs.
# Council.list_llms()

# # Initialize a council.
# council = Council(
#     "gpt-4.1",
#     "claude-sonnet-3.7",
#     "gemini-2.5-flash-001",
# )

# # Run the council on a prompt.
# completions_df = council.execute(prompt=prompt)

# # Council judging.
# judgments_df = council.judge(completions_df)

# # Produce analysis and visualizations for a specific session.
# # session[prompt].analyze()

# # Produce analysis and visualizations over all sessions.
# # council.analyze() # analyzes everything
# # council.analyze(prompt=prompt) # analyzes just this one prompt

# # Upload to HF datasets.
# council.upload_to_hf("<hf_username>", "<dataset_name>")


from llm_council.topologies.council import LanguageModelCouncil
from dotenv import load_dotenv
from llm_council.judging.schema import (
    EvaluationConfig,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_PAIRWISE_EVALUATION_CONFIG,
)

lmc = LanguageModelCouncil(
    models=[
        "google/gemini-2.5-flash-preview-05-20",
        "deepseek/deepseek-r1-0528",
        # "meta-llama/llama-3.1-8b-instruct",
    ],
    eval_config=DEFAULT_EVALUATION_CONFIG,
)

import asyncio


async def main():
    completions, judgments = await lmc.execute(prompt="Say hello.")
    completions, judgments = await lmc.execute(prompt="Say goodbye.")

    print(completions)
    print(judgments)

    print(lmc.get_completions_df())
    print(lmc.get_judgments_df())

    lmc.save("tests/testdata/sample_session")


asyncio.run(main())
