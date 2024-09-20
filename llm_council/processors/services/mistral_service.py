import dotenv
import logging
import os

from llm_council.processors.services import BaseService


dotenv.load_dotenv()


class MistralService(BaseService):
    """https://docs.mistral.ai/capabilities/completion/

    Sample curl command.

    curl https://api.mistral.ai/v1/chat/completions \
        --header "content-type: application/json" \
        --header "Accept: application/json" \
        --header "Authorization: Bearer $MISTRAL_API_KEY" \
        --data '{
        "model": "open-mistral-7b",
        "messages": [
        {
            "role": "user",
            "content": "What is the best French cheese?"
        }
        ]
    }'
    """

    def __init__(self, llm) -> None:
        BaseService.__init__(self, llm)

    def __api_key(self) -> str | None:
        return os.getenv("MISTRAL_API_KEY")

    def request_url(self) -> str:
        return "https://api.mistral.ai/v1/chat/completions"

    def request_header(self) -> dict:
        return {
            "content-type": "application/json",
            "Authorization": f"Bearer {self.__api_key()}",
            "Accept": "application/json",
        }

    def sample_request(self) -> dict:
        return {
            "model": "open-mistral-7b",
            "messages": [{"role": "user", "content": "Say hello!"}],
        }

    def rate_limit_time_unit(self) -> str:
        return "seconds"

    def max_requests_per_unit(self) -> int:
        return 5

    def is_response_error(self, response) -> bool:
        return "rate limit" in response["error"].get("message", "").lower()

    def get_request_prompt(self, request: dict) -> str:
        return request["messages"][0]["content"]

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: str | None
    ) -> dict:
        if schema_name is not None:
            logging.warning(
                f"Mistral does not support structured output. Skipping schema: {schema_name}."
            )
        if temperature is not None:
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": 1024,
                "temperature": temperature,
            }
        else:
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": 1024,
            }

    def get_response_string(self, json_response: dict) -> str:
        return json_response["choices"][0]["message"]["content"]

    def get_response_info(self, json_response: dict) -> dict:
        return {
            "llm": self.llm,
            "model_name": self.model_name,
            "id": json_response["id"],
            "usage": json_response["usage"],
        }
