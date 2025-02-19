import pandas as pd
from typing import List
from llm_council.members.membership import (
    LLM_TO_FULLY_QUALIFIED_NAME_MAP,
    FULLY_QUALIFIED_NAME_TO_LLM_MAP,
    LLMS,
)
from llm_council.providers.base_provider import PROVIDER_REGISTRY
from llm_council.members.schema import LanguageModel


def get_provider_from_fully_qualified_name(full_qualified_name):
    """Get the provider from the fully qualified name.

    openai://gpt-4-turbo-2024-04-09 -> openai
    """
    return full_qualified_name.split("://")[0]


def supported_providers() -> pd.DataFrame:
    """Return a DataFrame of enabled providers.

    Sample output:
                    api_key_name enabled
    anthropic  ANTHROPIC_API_KEY    True
    cerebras    CEREBRAS_API_KEY    True
    cohere        COHERE_API_KEY    True
    lepton        LEPTON_API_KEY    True
    mistral      MISTRAL_API_KEY    True
    openai        OPENAI_API_KEY    True
    together    TOGETHER_API_KEY    True
    vertex        VERTEX_API_KEY    True
    """
    df = pd.DataFrame(PROVIDER_REGISTRY).T
    # Drop the class column.
    df = df.drop(columns=["class"])
    return df


def supported_llms() -> pd.DataFrame:
    """Returns a DataFrame of supported LLMs.

    is_viable: True if the LLM is supported by at least one enabled provider.

    Sample output:
                Model Name is_viable anthropic cerebras cohere lepton mistral openai together vertex
    0   Qwen2-72B-Instruct         ✅         ❌        ❌      ❌      ✅       ❌      ❌        ✅      ❌
    1     Qwen1.5-72B-Chat         ✅         ❌        ❌      ❌      ❌       ❌      ❌        ✅      ❌
    2     Qwen1.5-32B-Chat         ✅         ❌        ❌      ❌      ❌       ❌      ❌        ✅      ❌
    3    Qwen1.5-110B-Chat         ✅         ❌        ❌      ❌      ❌       ❌      ❌        ✅      ❌
    4          gpt-4-turbo         ✅         ❌        ❌      ❌      ❌       ❌      ✅        ❌      ❌
    5                gpt-4         ✅         ❌        ❌      ❌      ❌       ❌      ✅        ❌      ❌
    6          gpt-4o-mini         ✅         ❌        ❌      ❌      ❌       ❌      ✅        ❌      ❌
    7               gpt-4o         ✅         ❌        ❌      ❌      ❌       ❌      ✅        ❌      ❌
    8           o1-preview         ✅         ❌        ❌      ❌      ❌       ❌      ✅        ❌      ❌
    9              o1-mini         ✅         ❌        ❌      ❌      ❌       ❌      ✅        ❌      ❌
    10      claude-3-haiku         ✅         ✅        ❌      ❌      ❌       ❌      ❌        ❌      ❌
    11     claude-3-sonnet         ✅         ✅        ❌      ❌      ❌       ❌      ❌        ❌      ❌
    12       claude-3-opus         ✅         ✅        ❌      ❌      ❌       ❌      ❌        ❌      ❌
    13   claude-3-5-sonnet         ✅         ✅        ❌      ❌      ❌       ❌      ❌        ❌      ❌
    14       mistral-large         ✅         ❌        ❌      ❌      ❌       ✅      ❌        ❌      ❌
    15       mixtral-8x22b         ✅         ❌        ❌      ❌      ❌       ✅      ❌        ❌      ❌
    16        mixtral-8x7b         ✅         ❌        ❌      ❌      ❌       ✅      ❌        ❌      ❌
    17      command-r-plus         ✅         ❌        ❌      ✅      ❌       ❌      ❌        ❌      ❌
    18      command-r-plus         ✅         ❌        ❌      ✅      ❌       ❌      ❌        ❌      ❌
    19      gemini-1.0-pro         ✅         ❌        ❌      ❌      ❌       ❌      ❌        ❌      ✅
    20      gemini-1.5-pro         ✅         ❌        ❌      ❌      ❌       ❌      ❌        ❌      ✅
    21    gemini-1.5-flash         ✅         ❌        ❌      ❌      ❌       ❌      ❌        ❌      ✅
    22        Llama-3.1-8B         ✅         ❌        ✅      ❌      ✅       ❌      ❌        ✅      ❌
    23       Llama-3.1-70B         ✅         ❌        ✅      ❌      ✅       ❌      ❌        ✅      ❌
    24      Llama-3.1-405B         ✅         ❌        ❌      ❌      ✅       ❌      ❌        ✅      ❌
    25       dbrx-instruct         ✅         ❌        ❌      ❌      ❌       ❌      ❌        ✅      ❌
    """
    # Extract unique provider names
    all_providers = set()
    for llm in LLMS:
        for provider in llm.providers:
            all_providers.add(provider.name)

    all_providers = sorted(all_providers)  # Ensure consistent ordering

    # Construct rows for the DataFrame
    rows = []
    for llm in LLMS:
        row = {"Model Name": llm.model_name}
        is_viable = False

        for provider in all_providers:
            # Check if this provider exists for the current model
            provider_instance = next(
                (p for p in llm.providers if p.name == provider), None
            )

            if provider_instance and PROVIDER_REGISTRY.get(provider, {}).get(
                "enabled", False
            ):
                row[provider] = "✅"
                is_viable = True
            else:
                row[provider] = "❌"

        row["is_viable"] = "✅" if is_viable else "❌"
        rows.append(row)

    # Reorder the rows such that is_viable is the first column
    columns = ["Model Name", "is_viable"] + all_providers
    return pd.DataFrame(rows)[columns]


# Run the functions on their own.
# print(supported_providers())
# print(supported_llms())
