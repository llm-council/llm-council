import os
import yaml
from pydantic import ValidationError
from collections import defaultdict
from llm_council.members.schema import LanguageModel


def parse_yaml_files(directory_path: str):
    language_models = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                try:
                    # Load the YAML file
                    yaml_content = yaml.safe_load(file)

                    # If the YAML file contains a list of models
                    if isinstance(yaml_content, list):
                        for model_data in yaml_content:
                            try:
                                # Parse each YAML entry into a LanguageModel object
                                language_model = LanguageModel(**model_data)
                                language_models.append(language_model)
                            except ValidationError as e:
                                print(f"Validation error in {filename}: {e}")
                    else:
                        # If it's a single model entry
                        language_model = LanguageModel(**yaml_content)
                        language_models.append(language_model)
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML file {filename}: {e}")

    return language_models


# Construct the path relative to the current file's location
current_dir = os.path.dirname(os.path.abspath(__file__))
LLMS = parse_yaml_files(current_dir)


def get_model_info(model_name: str):
    for llm in LLMS:
        if llm.model_name == model_name:
            return llm.model_info
    return None


def get_providers(model_name: str):
    for llm in LLMS:
        if llm.model_name == model_name:
            return llm.providers


def get_llm_to_fully_qualified_model_names_map() -> dict[str, list]:
    fully_qualified_model_names = defaultdict(list)
    for llm in LLMS:
        for provider in llm.providers:
            fully_qualified_model_names[llm.model_name].append(
                provider.fully_qualified_name
            )
    return fully_qualified_model_names


def get_fully_qualified_model_name_to_llm_map() -> dict[str, str]:
    fully_qualified_model_names = {}
    for llm in LLMS:
        for provider in llm.providers:
            fully_qualified_model_names[provider.fully_qualified_name] = llm.model_name
    return fully_qualified_model_names


# llm_short_name -> ["together://llm_name", "other_provider://llm_name_for_other_provider"]
LLM_TO_FULLY_QUALIFIED_NAME_MAP = get_llm_to_fully_qualified_model_names_map()

# "together://llm_name" -> llm_short_name
FULLY_QUALIFIED_NAME_TO_LLM_MAP = get_fully_qualified_model_name_to_llm_map()
