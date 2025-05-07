import dotenv
import logging
import os

import instructor
import openai
from llm_council.providers.base_provider import BaseProvider
from llm_council.structured_outputs import STRUCTURED_OUTPUT_REGISTRY
from llm_council.providers.base_provider import provider
from llm_council.judging.schema import EvaluationConfig
from llm_council.structured_outputs import create_dynamic_schema
from llm_council.providers.utils import (
    get_schema_class,
    check_prompt_template_contains_all_placeholders,
)
import importlib
import requests

dotenv.load_dotenv()


@provider(provider_name="openai", api_key_name="OPENAI_API_KEY")
class OpenAIProvider(BaseProvider):
    """https://platform.openai.com/docs/api-reference/making-requests"""

    def __init__(self, llm) -> None:
        BaseProvider.__init__(self, llm)

        self.async_client = openai.AsyncOpenAI()
        self.instructor_async_client = instructor.from_openai(self.async_client)

        if "gpt-4o-mini" in llm:
            self.max_requests_per_minute = 30000
        elif "gpt-4o" in llm:
            self.max_requests_per_minute = 10000
        elif "o1-preview" in llm:
            self.max_requests_per_minute = 500
        elif "o1-mini" in llm:
            self.max_requests_per_minute = 1000
        else:
            logging.warning(
                f"Unknown model for OpenAI Service: {llm}. Using default rate limit of 10K RPM."
            )
            self.max_requests_per_minute = 10000

    def __api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    def request_url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"

    def request_header(self) -> dict:
        return {"Authorization": f"Bearer {self.__api_key()}"}

    def sample_request(self) -> dict:
        return {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say hello!"}],
        }

    def rate_limit_time_unit(self) -> str:
        return "minutes"

    def max_requests_per_unit(self) -> int:
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
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema_class.schema(),
                },
            }
        return request

    def get_response_string(self, json_response: dict) -> str:
        return json_response["choices"][0]["message"]["content"]

    def get_response_info(self, json_response: dict) -> dict:
        return {
            "llm": self.llm,
            "model_name": self.model_name,
            "id": json_response["id"],
            "usage": json_response["usage"],
        }

    async def get_async_completion_task(
        self,
        prompt: str,
        task_metadata: dict,
        temperature: float | None = None,
        schema_class: type | None = None,
    ):
        """Perhaps this could also be shared across providers, as long as async_client and instructor_async_client are set.

        Default temperature is 1, and can be simply None:
            https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
        """
        if schema_class is not None:
            structured_output, completion = (
                await self.instructor_async_client.chat.completions.create_with_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_model=schema_class,
                )
            )
            return {
                "task_metadata": task_metadata,
                "structured_output": structured_output,
                "completion": completion,
            }
        else:
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return {
                "task_metadata": task_metadata,
                "completion_text": completion.choices[0].message.content,
                "completion": completion,
            }

    def list_models(self):
        def list_models(self):
            api_key = self.__api_key()
            if not api_key:
                raise ValueError("API key is not set.")

            url = "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {api_key}"}

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(
                    f"Failed to fetch models: {response.status_code}, {response.text}"
                )

            models = response.json().get("data", [])
            return [model["id"] for model in models]
