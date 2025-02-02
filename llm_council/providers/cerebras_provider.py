import dotenv
import os
import logging

from llm_council.providers.base_provider import BaseProvider
from llm_council.providers.base_provider import provider

dotenv.load_dotenv()


# Sample curl command.
# curl --location 'https://api.cerebras.ai/v1/chat/completions' \
# --header 'Content-Type: application/json' \
# --header "Authorization: Bearer CEREBRAS_API_KEY" \
# --data '{
#   "model": "llama3.1-8b",
#   "stream": false,
#   "messages": [{"content": "Hello!", "role": "user"}],
#   "temperature": 0,
#   "max_tokens": -1,
#   "seed": 0,
#   "top_p": 1
# }'
@provider(provider_name="cerebras", api_key_name="CEREBRAS_API_KEY")
class CerebrasProvider(BaseProvider):
    """https://inference-docs.cerebras.ai/introduction"""

    def __init__(self, llm) -> None:
        BaseProvider.__init__(self, llm)
        self.model_name = llm.split("://")[1]

    def __api_key(self) -> str:
        # https://cloud.cerebras.ai/platform/org_jve6rktvv63hfne69t6h8nwe/apikeys
        return os.getenv("CEREBRAS_API_KEY")

    def request_url(self) -> str:
        return "https://api.cerebras.ai/v1/chat/completions"

    def request_header(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key()}",
        }

    def sample_request(self) -> dict:
        return {
            "model": "llama3.1-8b",
            "stream": False,
            "messages": [{"content": "Say hello!", "role": "user"}],
            "temperature": 0,
            "max_tokens": -1,
            "seed": 0,
            "top_p": 1,
        }

    def rate_limit_time_unit(self) -> str:
        return "minute"

    def max_requests_per_unit(self) -> int:
        return 60

    def get_request_prompt(self, request: dict) -> str:
        return request["messages"][0]["content"]

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: str | None
    ) -> dict:
        if schema_name is not None:
            logging.warning(
                f"Cerebras does not support structured output. Skipping schema: {schema_name}."
            )
        if temperature is not None:
            return {
                "model": self.model_name,
                "stream": False,
                "messages": [{"content": user_prompt, "role": "user"}],
                "temperature": temperature,
                "max_tokens": -1,
            }

        return {
            "model": self.model_name,
            "stream": False,
            "messages": [{"content": user_prompt, "role": "user"}],
            "temperature": 0,
            "max_tokens": -1,
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
