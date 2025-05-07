import dotenv
import logging
import os
import instructor
import anthropic

from llm_council.providers.utils import get_schema_class
from llm_council.providers.base_provider import BaseProvider
from llm_council.providers.base_provider import provider

dotenv.load_dotenv()


@provider(provider_name="anthropic", api_key_name="ANTHROPIC_API_KEY")
class AnthropicProvider(BaseProvider):
    """https://docs.anthropic.com/en/api/messages"""

    def __init__(self, llm) -> None:
        BaseProvider.__init__(self, llm)

        # https://github.com/anthropics/anthropic-sdk-python
        self.async_client = anthropic.AsyncAnthropic()

        # Patching the Anthropics client with the instructor for enhanced capabilities
        self.instructor_async_client = instructor.from_anthropic(
            self.async_client,
            # create=anthropic.Anthropic().messages.create,
            # create=anthropic.AsyncAnthropic().messages.create,
            mode=instructor.Mode.ANTHROPIC_JSON,
        )

    def __api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    def request_url(self) -> str:
        return "https://api.anthropic.com/v1/messages"

    def request_header(self) -> dict:
        return {
            "x-api-key": self.__api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def sample_request(self) -> dict:
        return {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hello, world"}],
        }

    def rate_limit_time_unit(self) -> str:
        return "minutes"

    def max_requests_per_unit(self) -> int:
        return 2000

    def max_tokens_per_minute(self) -> int:
        return 200000

    def is_response_error(self, response) -> bool:
        return "rate-limits" in response["error"].get("message", "").lower()

    def get_request_user_prompt(self, request: dict) -> str:
        return request["messages"][0]["content"]

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: str | None
    ) -> dict:
        """Returns the data payload for a given user prompt."""
        if schema_name is not None:
            logging.warning(
                f"Anthropic does not support structured output. Skipping schema: {schema_name}."
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
        """Returns the model's response string."""
        return json_response["content"][0]["text"]

    def get_response_info(self, json_response: dict) -> dict:
        """Returns any relevant query information attached to the response."""
        return {
            "llm": self.llm,
            "model_name": self.model_name,
            "id": json_response["id"],
            "usage": json_response["usage"],
        }

    def get_request_prompt(self, request: dict) -> str:
        return request["messages"][0]["content"]

    async def get_async_completion_task(
        self,
        prompt: str,
        task_metadata: dict,
        temperature: float | None = None,
        schema_class: type | None = None,
        max_tokens: int = 2048,
    ):
        """Perhaps this could also be shared across providers, as long as async_client and instructor_async_client are set."""
        if schema_class is not None:
            # Anthropic SDK distinguishes arguments that are explicitly "not given" from those that are given as None.
            if temperature is not None:
                structured_output, completion = (
                    await self.instructor_async_client.chat.completions.create_with_completion(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        response_model=schema_class,
                        max_tokens=max_tokens,
                    )
                )
            else:
                structured_output, completion = (
                    await self.instructor_async_client.chat.completions.create_with_completion(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        response_model=schema_class,
                        max_tokens=max_tokens,
                    )
                )
            return {
                "task_metadata": task_metadata,
                "structured_output": structured_output,
                "completion": completion,
            }
        else:
            if temperature is not None:
                completion = await self.async_client.messages.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                completion = await self.async_client.messages.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                )
            return {
                "task_metadata": task_metadata,
                "completion_text": completion.content[0].text,
                "completion": completion,
            }
