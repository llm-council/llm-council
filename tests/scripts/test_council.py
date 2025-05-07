from llm_council.topologies.council import LanguageModelCouncil


if __name__ == "__main__":
    # Quick test.
    lmc = LanguageModelCouncil(
        llms=[
            "openai://gpt-4o-mini",
            "together://meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            # "anthropic://claude-3-haiku-20240307",
        ]
    )

    lmc.execute(prompt="Say hello.")

    lmc.save("tests/testdata/sample_session")
