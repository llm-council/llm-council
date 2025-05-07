import dotenv
import os
import logging
import openai
import instructor

from llm_council.providers.base_provider import BaseProvider
from llm_council.providers.base_provider import provider
from llm_council.structured_outputs import STRUCTURED_OUTPUT_REGISTRY
from llm_council.providers.utils import get_schema_class

dotenv.load_dotenv()


@provider(provider_name="together", api_key_name="TOGETHER_API_KEY")
class TogetherProvider(BaseProvider):
    """https://docs.together.ai/reference/chat-completions"""

    def __init__(self, llm) -> None:
        BaseProvider.__init__(self, llm)
        self.async_client = openai.AsyncOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY"),
        )
        self.instructor_async_client = instructor.from_openai(
            self.async_client, mode=instructor.Mode.TOOLS
        )

    def __api_key(self) -> str | None:
        return os.getenv("TOGETHER_API_KEY")

    def request_url(self) -> str:
        return "https://api.together.xyz/v1/chat/completions"

    def request_header(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key()}",
        }

    def sample_request(self) -> dict:
        return {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "messages": [{"role": "user", "content": "Say hello!"}],
        }

    def rate_limit_time_unit(self) -> str:
        return "seconds"

    def max_requests_per_unit(self) -> int:
        return 9

    def get_request_prompt(self, request: dict) -> str:
        return request["messages"][0]["content"]

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: str | None
    ) -> dict:
        if schema_name is not None:
            logging.warning(
                f"Together does not support structured output. Skipping schema: {schema_name}."
            )
        if temperature is not None:
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": temperature,
            }
        else:
            return {
                "model": self.model_name,
                "messages": [{"role": "user", "content": user_prompt}],
            }

    def get_response_string(self, json_response: dict) -> str:
        return json_response["choices"][0]["message"]["content"]

    def get_response_info(self, json_response: dict) -> dict:
        return {
            "llm": self.llm,
            "model_name": self.model_name,
            "id": json_response["id"],
            "usage": json_response["usage"],
            "model": json_response["model"],
        }

    async def get_async_completion_task(
        self,
        prompt: str,
        task_metadata: dict,
        temperature: float | None = None,
        schema_class: type | None = None,
    ):
        """Perhaps this could also be shared across providers, as long as async_client and instructor_async_client are set."""
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
