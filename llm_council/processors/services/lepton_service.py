import dotenv
import os

from leptonai.util import tool

from llm_council.processors.services.base_service import BaseService
from llm_council.structured_outputs import STRUCTURED_OUTPUT_REGISTRY

dotenv.load_dotenv()


class LeptonService(BaseService):
    """
    sample curl request:

    curl https://llama3-1-8b.lepton.run/api/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LEPTON_API_TOKEN" \
    -d '{
        "model": "llama3-1-8b",
        "messages": [{"role": "user", "content": "say hello"}],
        "temperature": 0.7
    }'
    """

    def __init__(self, llm) -> None:
        BaseService.__init__(self, llm)
        self.model_name = llm.split("://")[1]
        self.max_requests_per_minute = 10

    def __api_key(self):
        return os.getenv("LEPTON_API_KEY")

    def request_url(self):
        return f"https://{self.model_name}.lepton.run/api/v1/chat/completions"

    def request_header(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key()}",
        }

    def sample_request(self):
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Say hello!"}],
            "temperature": 0.7
        }

    def rate_limit_time_unit(self):
        return "minutes"

    def max_requests_per_unit(self):
        return self.max_requests_per_minute

    def max_tokens_per_minute(self) -> int:
        return 290000

    def get_request_prompt(self, request: dict) -> str:
        return request["messages"][0]["content"]

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: str | None
    ) -> dict:
        request = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if temperature is not None:
            request["temperature"] = temperature
        if schema_name is not None:
            schema_class = STRUCTURED_OUTPUT_REGISTRY.get(schema_name)
            if schema_class is None:
                raise ValueError(f"Invalid schema: {schema_name}")
            request["tools"] = [{
                "type": "function", 
                "function": tool.get_tools_spec(schema_class.method)
            }]

        return request

    def is_response_error(self, response) -> bool:
        return "quota exceeded" in response["error"].get("message", "").lower()

    def get_response_string(self, json_response: dict) -> str:
        return json_response["candidates"][0]["content"]["parts"][0]["text"]

    def get_response_info(self, json_response: dict) -> dict:
        return {
            "llm": self.llm,
            "model_name": self.model_name,
            "usage": json_response["usageMetadata"],
        }
