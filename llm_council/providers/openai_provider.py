import logging
import os

import instructor
import openai
from llm_council.providers.base_provider import BaseProvider
from llm_council.structured_outputs import STRUCTURED_OUTPUT_REGISTRY
from llm_council.providers.base_provider import provider


@provider(provider_name="openai", api_key_name="OPENAI_API_KEY")
class OpenAIProvider(BaseProvider):
    """https://platform.openai.com/docs/api-reference/making-requests"""

    def __init__(self, llm) -> None:
        BaseProvider.__init__(self, llm)

        self.instructor_async_client = instructor.from_openai(openai.AsyncOpenAI())
        self.async_client = openai.AsyncOpenAI()

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
        temperature: float | None = None,
        schema_name: str | None = None,
    ):
        if schema_name is not None:
            schema_class = STRUCTURED_OUTPUT_REGISTRY.get(schema_name)

            if temperature is None:
                return await self.instructor_async_client.chat.completions.create_with_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_model=schema_class,
                )
            else:
                return await self.instructor_async_client.chat.completions.create_with_completion(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_model=schema_class,
                )
        else:
            if temperature is None:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content, response
            else:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content, response
