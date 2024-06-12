"""Generate REST requests for the council based on an incoming jsonl file.

Assumes that the JSON objects from the incoming JSONL file can be passed as kwargs to the prompt
template, specified by `prompt_template_key`.

The JSON object is saved as metadata in written requests.
"""

import argparse
import jsonlines

from llm_council.utils import jsonl_io
from llm_council.prompts import get_realized_prompt, PROMPT_REGISTRY
from llm_council.processors.council_service import (
    get_default_council_service,
    CouncilService,
)


def generate_requests_for_response_generation(
    prompt_template_key: str,
    outdir: str,
    jsonl_input_file: str,
    word_limit: int,
    temperature: float | None,
    llm_allowlist: list[str] | None,
    split_requests_evenly: bool,
):
    if llm_allowlist is None:
        council_service = get_default_council_service(outdir)
    else:
        council_service = CouncilService(
            llm_council_members=llm_allowlist,
            outdir=outdir,
        )
    council_service.reset_request_files_for_council()

    if not split_requests_evenly:
        with jsonlines.open(jsonl_input_file) as reader:
            for json_obj in reader:
                if "id" not in json_obj.keys():
                    raise ValueError(
                        "JSON object must have an 'id' key because this is used to identify the request through the processing pipeline."
                    )
                realized_prompt = get_realized_prompt(
                    prompt_template_key, word_limit=word_limit, **json_obj
                )
                metadata = {
                    "completion_request": {
                        "prompt_template_key": prompt_template_key,
                        **json_obj,
                    }
                }
                council_service.write_council_request(
                    realized_prompt, metadata, temperature
                )
    else:
        # Read all json objects into memory.
        json_objs = []
        with jsonlines.open(jsonl_input_file) as reader:
            for json_obj in reader:
                json_objs.append(json_obj)

        # Split the requests evenly between council members.
        num_council_members = len(council_service.llm_council_members)
        for i, json_obj in enumerate(json_objs):
            if "id" not in json_obj.keys():
                raise ValueError(
                    "JSON object must have an 'id' key because this is used to identify the request through the processing pipeline."
                )
            realized_prompt = get_realized_prompt(
                prompt_template_key, word_limit=word_limit, **json_obj
            )
            metadata = {
                "completion_request": {
                    "prompt_template_key": prompt_template_key,
                    **json_obj,
                }
            }

            council_service.write_council_request_for_llm(
                llm=council_service.llm_council_members[i % num_council_members],
                prompt=realized_prompt,
                metadata=metadata,
                temperature=temperature,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate REST requests for the council based on the an incoming jsonl file.",
    )
    parser.add_argument(
        "--outdir",
        help="Path to the seed dilemma output file.",
        required=True,
    )
    parser.add_argument(
        "--jsonl_input_file",
        help="Path to dilemmas to use.",
        required=True,
    )
    parser.add_argument(
        "--prompt_template_key",
        help="The key of the prompt template to use.",
        required=True,
        choices=PROMPT_REGISTRY.keys(),
    )
    parser.add_argument(
        "--word_limit",
        help="The word limit for the generated prompt.",
        type=int,
        default=150,
        required=False,
    )
    parser.add_argument(
        "--temperature",
        help="The temperature to use when sampling from the model.",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--llm_allowlist",
        help="List of LLMs to allow for this request generation.",
        action="append",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--split_requests_evenly",
        help="Split requests evenly between council members.",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    generate_requests_for_response_generation(
        prompt_template_key=args.prompt_template_key,
        outdir=args.outdir,
        jsonl_input_file=args.jsonl_input_file,
        word_limit=args.word_limit,
        temperature=args.temperature,
        llm_allowlist=args.llm_allowlist,
        split_requests_evenly=args.split_requests_evenly,
    )
