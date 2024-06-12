"""Executes all of the requests in a given directory."""

import argparse
import json
import os
import time
import glob

from llm_council.utils import jsonl_io
from llm_council.processors.any_processor import run_processors_for_request_files


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

    start_time = time.time()
    request_files = jsonl_io.find_request_files(args.requests_dir)
    run_processors_for_request_files(request_files, args.requests_dir)
    end_time = time.time()

    print(f"Execution took {(end_time - start_time):.2f} seconds.")
