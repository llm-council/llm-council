"""Parses directory of `responses.jsonl` files. Outputs a consolidated JSONL file."""

import os
import argparse
import jsonlines
import csv
from collections import defaultdict
import pandas as pd
import re

from llm_council.utils import jsonl_io
from llm_council.processors.council_service import get_default_council_service


def truncate_to_sentence(text, word_limit):
    # Split the text into words to count them
    words = text.split()
    if len(words) <= word_limit:
        return text

    # Find sentences in the text
    sentences = re.split(r"(?<=[.!?]) +", text)

    # Rebuild the text sentence by sentence until the word limit is approached or exceeded
    truncated_text = ""
    word_count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if word_count + len(sentence_words) > word_limit:
            break
        truncated_text += sentence + " "
        word_count += len(sentence_words)

    # Check if truncating to the last full sentence removes more than half the words
    if word_count < word_limit / 2:
        return " ".join(words[:word_limit])  # Truncate directly to the word limit
    else:
        return truncated_text.strip()


def get_num_words(text):
    return len(text.split())


def consolidate_council_responses(
    council_response_metadata_key: str,
    responses_files: list[str],
    outdir: str,
    word_limit: int,
):
    consolidated_responses_path = os.path.join(outdir, "consolidated_responses.jsonl")
    jsonl_io.reset_file(consolidated_responses_path)

    council_service = get_default_council_service(outdir)
    responses_files = jsonl_io.find_response_files(args.responses_dir)

    llm_responder_to_num_words = defaultdict(list)
    llm_responder_to_truncation_amount = defaultdict(list)
    # The number of responses in each file should be the same.
    total_number_of_responses = 0
    with open(consolidated_responses_path, "w") as file:
        for responses_file in responses_files:
            response_counter = 0
            with jsonlines.open(responses_file) as reader:
                for response_data in reader:
                    response_counter += 1
                    llm_responder = response_data[-1]["llm"]
                    try:
                        response_string = council_service.get_llm_response_string(
                            response_data
                        )

                        request_prompt = council_service.get_llm_request_prompt(
                            response_data
                        )

                        # Enforce a word limit.
                        if word_limit:
                            llm_responder_to_num_words[llm_responder].append(
                                get_num_words(response_string)
                            )

                            maybe_truncated_response_string = truncate_to_sentence(
                                response_string, word_limit
                            )
                            llm_responder_to_truncation_amount[llm_responder].append(
                                get_num_words(response_string)
                                - get_num_words(maybe_truncated_response_string)
                            )

                            response_string = maybe_truncated_response_string

                        if not response_string:
                            breakpoint()
                    except:
                        # Some responses failed. Investigate.
                        print(
                            "\N{Angry Face}\N{Angry Face}\N{Angry Face}\N{Angry Face}\N{Angry Face}"
                            + f"Was not able to extract response from file: {responses_file}"
                        )
                        breakpoint()

                    metadata = response_data[-1]
                    metadata[council_response_metadata_key] = (
                        council_service.get_llm_response_query_info(response_data)
                    )
                    del metadata["llm"]
                    metadata["user_prompt"] = request_prompt
                    row = {
                        "id": metadata["completion_request"]["id"],
                        "user_prompt": request_prompt,
                        "llm_responder": llm_responder,
                        "response_string": response_string,
                        "metadata": metadata,
                    }

                    # Specifically for response completion requests.
                    if "context" in metadata["completion_request"]:
                        row["context"] = metadata["completion_request"]["context"]
                    jsonl_io.append_to_jsonl(row, consolidated_responses_path)

            if not total_number_of_responses:
                total_number_of_responses = response_counter
                print(f"Number of responses per file: {total_number_of_responses}")
            else:
                if total_number_of_responses != response_counter:
                    raise ValueError(
                        f"Number of responses is not consistent across files. Total number of responses from previous file(s) was: {total_number_of_responses}, but current file {responses_file} has: {response_counter} responses."
                    )

    # Write out truncation stats.
    if word_limit:
        num_words_stats = pd.DataFrame(llm_responder_to_num_words)
        print("-------------------------")
        print("Number of words stats:")
        print("-------------------------")
        print(num_words_stats.mean())
        num_words_stats.to_json(os.path.join(outdir, "num_words_stats.json"))

        print("-------------------------")

        truncation_amount = pd.DataFrame(llm_responder_to_truncation_amount)
        print("Truncation amount stats:")
        print("-------------------------")
        print(truncation_amount.mean())
        truncation_amount.to_json(os.path.join(outdir, "truncation_amount.json"))

    df = pd.read_json(consolidated_responses_path, lines=True, orient="records")

    df_pivoted = df.pivot_table(
        index="id",
        columns="llm_responder",
        values="response_string",
        aggfunc="first",
    )
    # As a CSV for human inpsection.
    df_pivoted.to_csv(os.path.join(outdir, "responses_for_inspection.csv"))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Parse responses JSONL files and output a CSV file"
    )
    parser.add_argument(
        "--responses_dir",
        type=str,
        help="The directory containing the responses to parse.",
    )
    parser.add_argument("--outdir", type=str, help="Path to the output directory.")
    parser.add_argument(
        "--council_response_metadata_key",
        type=str,
        help="Key to use for metadata in the consolidated responses.",
    )
    parser.add_argument(
        "--word_limit",
        type=int,
        help="The word limit for the generated prompt.",
        default=None,
        required=False,
    )

    args = parser.parse_args()

    consolidate_council_responses(
        args.council_response_metadata_key,
        args.responses_dir,
        args.outdir,
        args.word_limit,
    )
