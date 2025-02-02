import json
import os
from llm_council.providers.base_provider import PROVIDER_REGISTRY


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def get_provider_name(llm: str):
    return llm.split("://")[0]


def get_model_name(llm: str):
    return llm.split("://")[1]


def reset_file(filename: str) -> None:
    # Make the output directory if it does not exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        os.remove(filename)


def get_service_for_llm(llm: str):
    """Returns an instantiated service class for a given fully qualified LLM name."""

    provider_name = get_provider_name(llm)

    return PROVIDER_REGISTRY[provider_name]["class"](llm)
