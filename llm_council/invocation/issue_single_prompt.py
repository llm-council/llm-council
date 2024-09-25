"""Issues a single prompt to the LLM and returns the response.

python llm_council/invocation/issue_single_prompt.py \
    --prompt_file experiments/persuasiveness/one_shot.txt \
    --llm openai://gpt-3.5-turbo
"""

import argparse
import jsonlines
import os
import random
import sys
import time
from llm_council.processors.council_service import CouncilService

from llm_council.utils import jsonl_io
from llm_council.processors.any_processor import (
    run_processor_for_request_file,
)


def issue_single_prompt(prompt_file: str, llm: str):
    realized_prompt = open(prompt_file).read()

    request_output_file = args.prompt_file + ".request.jsonl"
    jsonl_io.reset_file(request_output_file)
    response_output_file = args.prompt_file + ".response.jsonl"
    jsonl_io.reset_file(response_output_file)
    raw_response_output_file = args.prompt_file + ".response.txt"
    jsonl_io.reset_file(raw_response_output_file)

    council_service = CouncilService(
        llm_council_members=[llm],
        outdir=os.path.dirname(args.prompt_file),
    )

    council_service.write_council_request(realized_prompt, {}, None, None)

    council_service.execute_council()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Issue a single prompt to the LLM and return the response.",
        description="Issue a single prompt to the LLM and return the response.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="The prompt file to issue to the LLM.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help="The LLM to issue the prompt to.",
    )

    args = parser.parse_args()

    issue_single_prompt(
        prompt_file=args.prompt_file,
        llm=args.llm,
    )
