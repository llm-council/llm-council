"""Script that takes in a JSONL file with bad responses, and reissues the requests to the LLMs that generated the bad responses.

python llm_council/invocation/rerun_bad_responses.py \
    --bad_responses_jsonl_file data_mmlu_lepton_100/mmlu_pro.n100.lepton.run2/so_jgt_cot1.reasoning_then_answer.temp1.student/lepton/llama3-1-8b/responses.jsonl
"""

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
