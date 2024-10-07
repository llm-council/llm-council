import argparse
import json
import jsonlines
import os

import pandas as pd

from llm_council.utils import jsonl_io
from llm_council.prompts import PROMPT_REGISTRY, get_realized_prompt
from llm_council.constants import LLM_COUNCIL_MEMBERS
from llm_council.processors.council_service import get_default_council_service
from llm_council.processors.council_service import (
    get_default_council_service,
    CouncilService,
)


def to_dict(series1, series2):
    return dict(zip(series1, series2))


def generate_requests_for_dsxs_judging(
    prompt_template_key: str,
    outdir: str,
    input_jsonl_file: str,
    temperature: float | None,
    num_reps: int,
    reference_llm: str,
    llm_allowlist: list[str] | None,
):
    # Create output directory if it does not exist.
    os.makedirs(outdir, exist_ok=True)

    if llm_allowlist is None:
        council_service = get_default_council_service(outdir)
    else:
        council_service = CouncilService(
            llm_council_members=llm_allowlist,
            outdir=outdir,
        )
    council_service.reset_request_files_for_council()

    df = pd.read_json(input_jsonl_file, lines=True, orient="records")
    id_to_llm_responder_to_response_string_dict = (
        df.groupby("id")[["llm_responder", "response_string"]]
        .apply(lambda x: to_dict(x["llm_responder"], x["response_string"]))
        .to_dict()
    )
    id_to_llm_responder_to_metadata_dict = (
        df.groupby("id")[["llm_responder", "metadata"]]
        .apply(lambda x: to_dict(x["llm_responder"], x["metadata"]))
        .to_dict()
    )

    # Get all pairs of responses to judge.
    all_pairs = []
    for id in id_to_llm_responder_to_response_string_dict.keys():
        for llm in id_to_llm_responder_to_response_string_dict[id].keys():
            if llm == reference_llm:
                # Skip reference vs. reference.
                continue
            if (
                "context"
                in id_to_llm_responder_to_metadata_dict[id][llm]["completion_request"]
            ):
                context = id_to_llm_responder_to_metadata_dict[id][llm][
                    "completion_request"
                ]["context"]
            else:
                # context = "synthetic"
                # context = id_to_llm_responder_to_metadata_dict[id][llm][
                #     "completion_request"
                # ]["response_string"]
                context = id_to_llm_responder_to_metadata_dict[id][llm]["user_prompt"]
            all_pairs.append(
                {
                    "first_completion": id_to_llm_responder_to_response_string_dict[id][
                        llm
                    ],
                    "second_completion": id_to_llm_responder_to_response_string_dict[
                        id
                    ][reference_llm],
                    "first_completion_by": llm,
                    "second_completion_by": reference_llm,
                    "metadata": id_to_llm_responder_to_metadata_dict[id][llm],
                    "context": context,
                }
            )
            all_pairs.append(
                {
                    "second_completion": id_to_llm_responder_to_response_string_dict[
                        id
                    ][llm],
                    "first_completion": id_to_llm_responder_to_response_string_dict[id][
                        reference_llm
                    ],
                    "second_completion_by": llm,
                    "first_completion_by": reference_llm,
                    "metadata": id_to_llm_responder_to_metadata_dict[id][llm],
                    "context": context,
                }
            )

    for _ in range(num_reps):
        # Generate requests for each pair.
        for pair in all_pairs:
            realized_prompt = get_realized_prompt(prompt_template_key, **pair)
            metadata = pair["metadata"]
            metadata["judging_request"] = {
                "first_completion_by": pair["first_completion_by"],
                "second_completion_by": pair["second_completion_by"],
            }
            council_service.write_council_request(
                realized_prompt, metadata, temperature
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Generate LLM requests to judge LLM responses.",
        description="Generate LLM requests to judge LLM responses.",
    )
    parser.add_argument(
        "--input_jsonl_file", help="Path to LLM responses", required=True
    )
    parser.add_argument(
        "--outdir",
        help="Output directory.",
        required=True,
    )
    parser.add_argument(
        "--prompt_template_key",
        help="Prompt template key",
        required=True,
        choices=PROMPT_REGISTRY.keys(),
    )
    parser.add_argument(
        "--temperature",
        help="The temperature to use when sampling from the model.",
        type=float,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--num_reps",
        help="The number of repetitions for each request.",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--reference_llm",
        help="The reference LLM respondent to compare against.",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--llm_allowlist",
        help="List of LLMs to allow for this request generation.",
        action="append",
        default=None,
        required=False,
    )

    main_args = parser.parse_args()

    generate_requests_for_dsxs_judging(
        prompt_template_key=main_args.prompt_template_key,
        outdir=main_args.outdir,
        input_jsonl_file=main_args.input_jsonl_file,
        temperature=main_args.temperature,
        num_reps=main_args.num_reps,
        reference_llm=main_args.reference_llm,
        llm_allowlist=main_args.llm_allowlist,
    )
