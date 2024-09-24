"""Utility functions to help with the structure-specific interfacing to/from JSONL artifacts."""

import json
import os
import glob


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def reset_file(filename: str) -> None:
    # Make the output directory if it does not exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        os.remove(filename)


def find_request_files(base_dir: str) -> list[str]:
    """
    Searches recursively for all files named 'requests.jsonl' in the given directory.

    Args:
    base_dir (str): The directory to search within.

    Returns:
    list[str]: A list of full paths to each 'requests.jsonl' file found.
    """
    # Create the pattern to search for
    pattern = "**/requests.jsonl"
    # Use glob to find all paths that match the pattern, recursively
    file_paths = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    return file_paths


def find_response_files(base_dir: str) -> list[str]:
    """
    Searches recursively for all files named 'responses.jsonl' in the given directory.

    Args:
    base_dir (str): The directory to search within.

    Returns:
    list[str]: A list of full paths to each 'responses.jsonl' file found.
    """
    # Create the pattern to search for
    pattern = "**/responses.jsonl"
    # Use glob to find all paths that match the pattern, recursively
    file_paths = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    return file_paths
