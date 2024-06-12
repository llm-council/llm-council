"""GENERIC REST-BASED PARALLEL PROCESSOR

Using serverless APIs to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to a serverless endpoint while throttling to stay under rate limits.
"""

from llm_council.processors.services import BaseService
from llm_council.utils.jsonl_io import append_to_jsonl

import requests as rq
import os
import aiohttp
import argparse
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
import dotenv
import tiktoken


def num_tokens_consumed_from_request(
    request_json: dict,
    service_config: BaseService,
):
    """Count the number of tokens in the request."""
    num_tokens = 0
    # Default to gpt-3.5-turbo encoding.
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # tokens = prompt + (n * max_tokens), default to 200.
    max_tokens = request_json.get("max_tokens", 200)
    n = request_json.get("n", 1)
    completion_tokens = n * max_tokens

    prompt = service_config.get_request_prompt(request_json)
    num_tokens += len(encoding.encode(prompt))
    return num_tokens + completion_tokens


# For use with asyncio.run.
async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    service_config: BaseService,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = (
        service_config.seconds_to_pause_after_rate_limit_error()
    )
    seconds_to_sleep_each_loop = service_config.seconds_to_sleep_each_loop()

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    request_header = service_config.request_header()
    request_url = service_config.request_url()
    max_requests_per_unit = service_config.max_requests_per_unit()
    max_tokens_per_minute = service_config.max_tokens_per_minute()

    # Say hello, and raise an exception if it fails.
    try:
        rq.post(
            request_url,
            service_config.sample_request(),
            headers=request_header,
        )
    except Exception as e:
        print(e)
        print("Saying hello failed, exiting.")
        raise

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_unit
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, service_config
                                ),
                                attempts_left=service_config.max_attempts(),
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time

                if service_config.rate_limit_time_unit() == "minutes":
                    available_request_capacity = min(
                        available_request_capacity
                        + max_requests_per_unit * seconds_since_update / 60.0,
                        max_requests_per_unit,
                    )
                else:
                    available_request_capacity = min(
                        available_request_capacity
                        + max_requests_per_unit * seconds_since_update,
                        max_requests_per_unit,
                    )

                # Update token capacity, if applicable.
                if max_tokens_per_minute:
                    available_token_capacity = min(
                        available_token_capacity
                        + max_tokens_per_minute * seconds_since_update / 60.0,
                        max_tokens_per_minute,
                    )

                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if available_request_capacity >= 1 and (
                        available_token_capacity is None
                        or available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        if available_token_capacity:
                            available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                                service_config=service_config,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
        service_config,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if service_config.is_response_error(response):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [
                        {"llm": service_config.llm},
                        self.request_json,
                        [str(e) for e in self.result],
                        self.metadata,
                    ]
                    if self.metadata
                    else [
                        {"llm": service_config.llm},
                        self.request_json,
                        [str(e) for e in self.result],
                    ]
                )
                append_to_jsonl(data, save_filepath.replace(".jsonl", "_errors.jsonl"))
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [
                    {"llm": service_config.llm},
                    self.request_json,
                    response,
                    self.metadata,
                ]
                if self.metadata
                else [
                    {"llm": service_config.llm},
                    self.request_json,
                    response,
                ]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1
