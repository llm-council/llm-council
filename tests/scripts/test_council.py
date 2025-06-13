from llm_council.topologies.council import LanguageModelCouncil


if __name__ == "__main__":
    # Quick test.
    lmc = LanguageModelCouncil(
        models=[
            "google/gemini-2.5-flash-preview-05-20",
            # "deepseek/deepseek-r1-0528",
            "meta-llama/llama-3.1-8b-instruct",
        ]
    )

    # completions, judgements = await lmc.execute_notebook(prompt="Say hello.")

    lmc.execute(prompt="Say hello.")

    lmc.save("tests/testdata/sample_session")
