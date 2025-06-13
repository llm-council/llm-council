from llm_council.topologies.council import LanguageModelCouncil


if __name__ == "__main__":
    # Quick test.
    lmc = LanguageModelCouncil(
        models=[
            "google/gemini-2.5-flash-preview-05-20",
            "meta-llama/llama-3.1-8b-instruct",
        ]
    )

    lmc.execute(prompt="Say hello.")

    lmc.save("tests/testdata/sample_session")
    lmc.load("tests/testdata/sample_session")

    assert lmc.get_completions_df().shape[0] == 2, "No completions found."
