import dotenv
import os
import logging

from llm_council.providers.base_provider import BaseProvider
from llm_council.providers.base_provider import provider

dotenv.load_dotenv()


@provider(provider_name="cohere", api_key_name="COHERE_API_KEY")
class CohereProvider(BaseProvider):
    """https://docs.cohere.com/reference/chat"""

    def __init__(self, llm) -> None:
        BaseProvider.__init__(self, llm)

    def __api_key(self) -> str | None:
        return os.getenv("COHERE_API_KEY")

    def request_url(self) -> str:
        return "https://api.cohere.ai/v1/chat"

    def request_header(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key()}",
        }

    def sample_request(self) -> dict:
        return {
            "model": "command-r",
            "message": "Say hello!",
        }

    def rate_limit_time_unit(self) -> str:
        return "seconds"

    def max_requests_per_unit(self) -> int:
        return 95

    def get_request_prompt(self, request: dict) -> str:
        return request["message"]

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: str | None
    ) -> dict:
        if schema_name is not None:
            logging.warning(
                f"Cohere does not support structured output. Skipping schema: {schema_name}."
            )
        if temperature is not None:
            return {
                "model": self.model_name,
                "message": user_prompt,
                "temperature": temperature,
            }
        else:
            return {
                "model": self.model_name,
                "message": user_prompt,
            }

    def get_response_string(self, json_response: dict) -> str:
        return json_response["text"]

    def get_response_info(self, json_response: dict) -> dict:
        return {
            "llm": self.llm,
            "model_name": self.model_name,
            "response_id": json_response["response_id"],
            "generation_id": json_response["generation_id"],
            "finish_reason": json_response["finish_reason"],
            "meta": json_response["meta"],
        }
