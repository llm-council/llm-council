"""A Provider powers a Service.

A Service defines a specification that is used by the Processor to manage large volumes of requests and responses for an LLM.
- It defines request format (header and body), response extraction, and query information.
- It defines rate limits, api keys, and request urls.

The Council Service consists of multiple LLMs, which are attended by their own Service.
"""
import dotenv
import os

from llm_council.processors.services.utils import get_model_name, get_provider_name, reset_file


dotenv.load_dotenv()


class BaseService:

    def __init__(self, llm: str) -> None:
        self.llm = llm
        self.model_name = get_model_name(llm)
        self.provider_name = get_provider_name(llm)
        pass

    def __api_key(self) -> str | None: 
        raise NotImplementedError

    def request_url(self) -> str:
        raise NotImplementedError

    def request_header(self) -> dict:
        raise NotImplementedError

    def sample_request(self) -> dict:
        raise NotImplementedError

    def rate_limit_time_unit(self) -> str:
        # "minutes", "seconds"
        raise NotImplementedError

    def max_requests_per_unit(self) -> int:
        raise NotImplementedError

    def max_tokens_per_minute(self) -> int:
        raise NotImplementedError

    def get_request_prompt(self, request: dict) -> str:
        """Returns the user prompt in the request body."""
        raise NotImplementedError

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: str | None
    ) -> dict:
        """Returns the data payload for a given user prompt."""
        raise NotImplementedError

    def get_response_string(self, json_response: dict) -> str:
        """Returns the model's response string."""
        raise NotImplementedError

    def get_response_info(self, json_response: dict) -> dict:
        """Returns any relevant query information attached to the response."""
        raise NotImplementedError

    def seconds_to_sleep_each_loop(self) -> float:
        # 1 ms limits max throughput to 1,000 requests per second
        return 0.001

    def seconds_to_pause_after_rate_limit_error(self) -> int:
        return 15

    def max_attempts(self) -> int:
        return 5

    def get_llm(self) -> str:
        return self.llm

    def is_response_error(self, response) -> bool:
        """Determines if the original request should be retried due to a rate limit error,
        potentially with a cooldown period.
        """
        return "Rate limit" in response["error"].get("message", "")

    def get_requests_path(self, outdir: str):
        directory = os.path.join(outdir, self.provider_name, self.model_name)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, "requests.jsonl")

    def get_responses_path(self, outdir: str):
        directory = os.path.join(outdir, self.provider_name, self.model_name)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, "responses.jsonl")

    def reset_requests(self, outdir: str):
        reset_file(self.get_requests_path(outdir))

    def reset_requests_and_responses(self, outdir: str):
        reset_file(self.get_requests_path(outdir))
        reset_file(self.get_responses_path(outdir))
