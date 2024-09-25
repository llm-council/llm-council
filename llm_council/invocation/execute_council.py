"""Executes all of the requests in a given directory."""
import argparse
import time


from llm_council.utils import jsonl_io
from llm_council.processors.any_processor import run_processors_for_request_files


def execute(requests_dir: str) -> None:
    start_time = time.time()
    request_files = jsonl_io.find_request_files(requests_dir)
    run_processors_for_request_files(request_files, requests_dir)
    end_time = time.time()

    print(f"Execution took {(end_time - start_time):.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Execute all requests in a directory.",
        description="Execute all requests in a directory.",
    )
    parser.add_argument(
        "--requests_dir",
        type=str,
        required=True,
        help="The directory containing the requests to execute.",
    )
    args = parser.parse_args()
    
    execute(args.requests_dir)
