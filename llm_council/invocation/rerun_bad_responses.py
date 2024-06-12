"""Script that takes in a JSONL file with bad responses, and reissues the requests to the LLMs that generated the bad responses."""

import argparse
from llm_council.processors.any_processor import rerun_bad_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Rerun bad responses.",
        description="Script that takes in a JSONL file with bad responses, and reissues the requests to the LLMs that generated the bad responses.",
    )
    parser.add_argument(
        "--bad_responses_jsonl_file",
        help="Path to the JSONL file with bad responses.",
        required=True,
    )
    parser.add_argument(
        "--run_all",
        help="If set, rerun all responses, not just the bad ones.",
        default=False,
    )

    args = parser.parse_args()

    rerun_bad_responses(args.bad_responses_jsonl_file, args.run_all)
