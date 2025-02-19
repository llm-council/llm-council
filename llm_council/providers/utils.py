import json
import re
import os
from llm_council.members.membership import LLM_TO_FULLY_QUALIFIED_NAME_MAP
from llm_council.judging.prompt_builder import DIRECT_ASSESSMENT_SPECIAL_PLACEHOLDERS
import string
import importlib


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def get_provider_name(llm: str, allowed_providers=None):
    if "://" in llm:
        # If the provider is specified in the LLM fully qualified path, use it.
        return llm.split("://")[0]
    else:
        # Look up the provider from the registry.
        if llm not in LLM_TO_FULLY_QUALIFIED_NAME_MAP:
            raise ValueError(f"LLM {llm} not found in LLM_TO_FULLY_QUALIFIED_NAME_MAP.")
        fully_qualified_names = LLM_TO_FULLY_QUALIFIED_NAME_MAP[llm]
        if allowed_providers is None:
            return fully_qualified_names[0].split("://")[0]
        else:
            for fully_qualified_name in fully_qualified_names:
                provider_name = fully_qualified_name.split("://")[0]
                if provider_name in allowed_providers:
                    return provider_name
            raise ValueError(
                f"None of the providers for LLM {llm} ({fully_qualified_names}) are in the allowed providers list: {allowed_providers}."
            )


def get_model_name(llm: str):
    if "://" in llm:
        # If the provider is specified in the LLM fully qualified path, use it.
        return llm.split("://")[1]
    return llm


def reset_file(filename: str) -> None:
    # Make the output directory if it does not exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        os.remove(filename)


def get_placeholders(s):
    # formatter = string.Formatter()
    # return [field_name for field_name, *_ in formatter.parse(s) if field_name]
    return re.findall(r"\{([^}]+)\}", s)


def check_prompt_template_contains_all_placeholders(prompt_template, prompt_fields):
    prompt_placeholders = get_placeholders(prompt_template)
    metadata_keys = list(prompt_fields.keys()) + DIRECT_ASSESSMENT_SPECIAL_PLACEHOLDERS
    if not set(prompt_placeholders).issubset(metadata_keys):
        raise ValueError(
            f"Placeholders not accounted for: {set(prompt_placeholders) - set(metadata_keys)}."
        )


def get_schema_class(schema_class_path):
    """Imports the schema class from the given path and returns the class definition."""
    if schema_class_path is not None:
        module_path, class_name = schema_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
