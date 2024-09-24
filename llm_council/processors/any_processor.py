import asyncio
import jsonlines
import logging
import os
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_council.processors import generic_processor
from llm_council.utils import jsonl_io
from llm_council.processors.services import PROVIDER_REGISTRY


def get_llm(request_file):
    """Get the LLM from the request file."""
    with jsonlines.open(request_file) as reader:
        for request in reader:
            return request["metadata"]["llm"]


def get_service_name(llm):
    return llm.split("://")[0]


def count_lines_in_file(filename: str):
    """Count the number of lines in a given file."""
    with open(filename, "r") as file:
        return len(file.readlines())


def log_and_write_timing(start_time, end_time, outdir):
    total_time = end_time - start_time
    time_log_str = f"Finished in: {total_time:.2f}(s)."

    print(time_log_str)
    with open(os.path.join(outdir, "timing.log"), "w") as f:
        f.write(time_log_str)


def run_processor_for_request_file(request_file: str, outfile: str):
    llm = get_llm(request_file)
    service_name = get_service_name(llm)
    service_instance = PROVIDER_REGISTRY[service_name](llm)

    start_time = time.time()
    asyncio.run(
        generic_processor.process_api_requests_from_file(
            request_file,
            outfile,
            service_config=service_instance,
            logging_level=logging.INFO,
        )
    )
    log_and_write_timing(start_time, time.time(), os.path.dirname(outfile))


def group_files_by_subdir(paths, directory=None):
    if not paths:
        return {}

    if not directory:
        # Split each path into parts
        split_paths = [path.split("/") for path in paths]

        # Assume the minimum length path is the common path initially
        min_length = min(len(p) for p in split_paths)

        common_path_length = 0

        # Determine the length of the longest common path
        for i in range(min_length):
            current_part = split_paths[0][i]
            if all(p[i] == current_part for p in split_paths):
                common_path_length += 1
            else:
                break
    else:
        common_path_length = len(directory.split("/"))

    # Group files by the first directory after the common path
    grouped_files = {}
    for path in paths:
        parts = path.split("/")

        # If there is a directory after the common path, use it as a key
        if len(parts) > common_path_length:
            subdir = parts[common_path_length]
            if subdir not in grouped_files:
                grouped_files[subdir] = []
            grouped_files[subdir].append(path)

    return grouped_files


def run_processors_for_request_files(request_files: list[str], directory: str):
    """Runs processors for each dict in parallel using threading."""
    # Group files by the first directory after the common path.
    request_files_grouped_by_provider = defaultdict(list)
    request_file_to_num_lines = {}
    provider_to_num_lines = defaultdict(int)
    for request_file in request_files:
        llm = get_llm(request_file)
        provider = get_service_name(llm)
        request_files_grouped_by_provider[provider].append(request_file)
        num_lines_in_request_file = count_lines_in_file(request_file)
        request_file_to_num_lines[request_file] = num_lines_in_request_file
        provider_to_num_lines[provider] += num_lines_in_request_file

    print("#" * 30)
    print("Parallelized request file groups:")
    print("#" * 30)
    for provider, request_files_group in request_files_grouped_by_provider.items():
        print("-" * 10)
        print(
            f"{provider}: {len(request_files_group)} file(s) with {provider_to_num_lines[provider]} requests."
        )
        for request_file in request_files_group:
            print(
                f"  - {request_file} ({request_file_to_num_lines[request_file]} requests)"
            )
    print("-" * 10)
    print(f"{sum(provider_to_num_lines.values())} total requests over all providers.")

    if not get_confirmation():
        print("Operation cancelled.")
        return

    print("Starting parallelized processing.")

    def process_file(request_files: list[str]):
        for request_file in request_files:
            outfile = os.path.join(os.path.dirname(request_file), "responses.jsonl")
            jsonl_io.reset_file(outfile)

            print(f"Collecting responses for {request_file}.")
            run_processor_for_request_file(request_file, outfile)
            print(
                f"""\N{Party Popper} Finished collecting for {request_file}. \N{Party Popper}"""
            )

    # Use ThreadPoolExecutor to run tasks in parallel
    with ThreadPoolExecutor() as executor:
        # Dictionary to hold future tasks
        futures = [
            executor.submit(process_file, request_files_group)
            for request_files_group in request_files_grouped_by_provider.values()
        ]
        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()  # Retrieve the result to catch any exceptions
            except Exception as exc:
                print(
                    f"An exception was encountered while running a request file group: {exc}"
                )


def get_confirmation():
    response = input("Do you want to proceed? [Y/n]: ")
    return response.lower() in [
        "yes",
        "y",
        "",
    ]  # '' represents the default "yes" when Enter is pressed


def get_serverless_provider(llm: str):
    if llm.startswith("openai://"):
        return "openai"
    if llm.startswith("hf://"):
        return "hf"
    if llm.startswith("google://"):
        return "google"
    if llm.startswith("anthropic://"):
        return "anthropic"
    if llm.startswith("mistral://"):
        return "mistral"
    if llm.startswith("together://"):
        return "together"
    if llm.startswith("cohere://"):
        return "cohere"
    if llm.startswith("vertex://"):
        return "vertex"
    raise ValueError(f"Unknown serverless LLM provder: '{llm}.")


def get_llm_response_string(json_response: dict) -> str:
    """Get the response from an LLM response."""
    llm = json_response[0]["llm"]
    provider = get_serverless_provider(llm)
    service = PROVIDER_REGISTRY[provider](llm)
    return service.get_response_string(json_response[2])


def rerun_bad_responses(responses_file: str, run_all: bool):
    """Rerun bad responses in a responses file. Pass in run_all=True to rerun all responses."""
    bad_responses = []
    good_responses = []
    with jsonlines.open(responses_file) as reader:
        for response in reader:
            if run_all:
                bad_responses.append(response)
                continue
            try:
                # Logic to find bad responses.
                response_string = get_llm_response_string(response)
                if not response_string:
                    bad_responses.append(response)
                elif "rate-limit" in response_string:
                    bad_responses.append(response)
                elif "Apologies," in response_string:
                    bad_responses.append(response)
                else:
                    good_responses.append(response)
            except:
                bad_responses.append(response)
                continue
    print(f"Found {len(bad_responses)} bad responses.")
    if len(bad_responses) == 0:
        print("No bad responses found.")
        return

    # Ask for user confirmation before proceeding.
    if not get_confirmation():
        print("Operation cancelled.")
        return

    print("Rerunning bad responses.")

    # Write bad responses to an extra file of additional requests.
    requests_to_rerun_path = os.path.join(
        os.path.dirname(responses_file), "extra_requests.jsonl"
    )
    jsonl_io.reset_file(requests_to_rerun_path)
    for response in bad_responses:
        # Turn the response back into its original request.
        request = response[1]
        request["metadata"] = response[-1]
        jsonl_io.append_to_jsonl(request, requests_to_rerun_path)

    # Run the processor for the requests that had bad responses.
    responses_from_rerun_requests_path = os.path.join(
        os.path.dirname(responses_file), "extra_responses.jsonl"
    )
    jsonl_io.reset_file(responses_from_rerun_requests_path)

    run_processor_for_request_file(
        requests_to_rerun_path, responses_from_rerun_requests_path
    )

    # Clear the original responses file.
    if os.path.exists(responses_file):
        os.remove(responses_file)

    print(f"Extra requests written to: {requests_to_rerun_path}")
    print(f"Extra responses written to: {responses_from_rerun_requests_path}")

    # Merge the good responses with the rerun responses.
    with jsonlines.open(responses_file, "w") as writer:
        for response in good_responses:
            writer.write(response)
        with jsonlines.open(responses_from_rerun_requests_path) as reader:
            for response in reader:
                writer.write(response)

    print(f"Merged responses rewritten to: {responses_file}")
