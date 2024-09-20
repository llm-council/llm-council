import os
import logging

from llm_council.processors.services import BaseService


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
        return f"https://{self.model_name}.lepton.run/api/v1/"

    def request_header(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key()}",
        }

    def sample_request(self):
        return {
            "model": self.model_name,
            "messages": [
                {"role": "user", "parts": {"text": "Say hello!"}},
            ],
            "temperature": 0.7
        }

    def rate_limit_time_unit(self):
        return "minutes"

    def max_requests_per_unit(self):
        return self.max_requests_per_minute

    def get_request_prompt(self, request: dict) -> str:
        return request["contents"][0]["parts"][0]["text"]

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema: str | None
    ) -> dict:
        if schema is not None:
            logging.warning(
                f"Vertex does not support structured output. Skipping schema: {schema}."
            )
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]
        if temperature is not None:
            return {
                "contents": [
                    {
                        "role": "USER",
                        "parts": [
                            {
                                "text": user_prompt,
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                },
                "safetySettings": safety_settings,
            }
        else:
            return {
                "contents": [
                    {
                        "role": "USER",
                        "parts": [
                            {
                                "text": user_prompt,
                            }
                        ],
                    }
                ],
                "safetySettings": safety_settings,
            }

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
