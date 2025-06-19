import pytest
from llm_council.council import LanguageModelCouncil
import dotenv
import shutil


@pytest.mark.asyncio
async def test_language_model_council():
    dotenv.load_dotenv()

    lmc = LanguageModelCouncil(
        models=[
            "google/gemini-2.5-flash-preview-05-20",
            "meta-llama/llama-3.1-8b-instruct",
        ]
    )

    await lmc.execute(prompt="Say hello.")

    lmc.save("tests/testdata/sample_session")
    lmc.load("tests/testdata/sample_session")
    shutil.rmtree("tests/testdata/sample_session")

    assert lmc.get_completions_df().shape[0] == 2, "No completions found."
