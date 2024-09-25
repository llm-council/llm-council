"""A Provider powers a Service.

A Service defines a specification that is used by the Processor to manage large volumes of requests and responses for an LLM.
- It defines request format (header and body), response extraction, and query information.
- It defines rate limits, api keys, and request urls.

The Council Service consists of multiple LLMs, which are attended by their own Service.
"""

import dotenv
import os
from llm_council.structured_outputs import STRUCTURED_OUTPUT_REGISTRY


dotenv.load_dotenv()


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def get_provider_name(llm: str):
    return llm.split("://")[0]


def get_model_name(llm: str):
    return llm.split("://")[1]


def reset_file(filename: str) -> None:
    # Make the output directory if it does not exist.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        os.remove(filename)


class BaseService:

    def __init__(self, llm: str) -> None:
        self.llm = llm
        self.model_name = get_model_name(llm)
        self.provider_name = get_provider_name(llm)
        pass

    def __api_key(self) -> str:
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
        return None

    def get_request_prompt(self, request: dict) -> str:
        """Returns the user prompt in the request body."""
        raise NotImplementedError

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: dict | None
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


class OpenAIService(BaseService):
    """https://platform.openai.com/docs/api-reference/making-requests"""

    def __init__(self, llm) -> None:
        BaseService.__init__(self, llm)

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

    def __api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY")

    def request_url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"

    def request_header(self) -> dict:
        return {"Authorization": f"Bearer {self.__api_key()}"}

    def sample_request(self) -> dict:
        return {
            "model": "gpt-3.5-turbo-0613",
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


class AnthropicService(BaseService):
    """https://docs.anthropic.com/en/api/messages"""

    def __init__(self, llm) -> None:
        BaseService.__init__(self, llm)

    def __api_key(self) -> str:
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


class CohereService(BaseService):
    """https://docs.cohere.com/reference/chat"""

    def __init__(self, llm) -> None:
        BaseService.__init__(self, llm)

    def __api_key(self) -> str:
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


class TogetherService(BaseService):
    """https://docs.together.ai/reference/chat-completions"""

    def __init__(self, llm) -> None:
        BaseService.__init__(self, llm)

    def __api_key(self) -> str:
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

    def __api_key(self) -> str:
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


class VertexService(BaseService):
    """https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini?_gl=1*uk6ij0*_ga*MTkwMTgxNzc1MS4xNzE0MDc0OTM4*_ga_WH2QY8WWF5*MTcxNTg0MjkwNS4zOS4xLjE3MTU4NDI5MjguMC4wLjA.&_ga=2.185445036.480671911.1715842905-1901817751.1714074938
    
    Sample curl command.

curl \
-X POST \
-H "Authorization: Bearer ya29.a0AcM612x8pWzLSlmukD_j524NBX3uoLgmi80tI7tVGdBQuVBYojEqiBgtJViZ0e4i135ILGE0FWAKPypbULyf3mzONKrWMfQ6Ihw15sfbglz1Q18JubeTw4hecunk0Zg4B6iXnu2ZGvHL_0LfXbBI6vy3lPC09Y5Pjj1RQXXST69ZMFCUUMAMiQpCrT2awn5Aa9XJDVJ2PEGTjQto_Stl9T1wn4bN0QDUjUw4NU0VvpR3ZTJib9AWhZB4cKGMyyd_apPca4MmdoErLy5crLZgCZrRtrUe0HDhQTcTGFl2WEh7SxsZ5xVWY90p2Ay5MHqRw6-5XRc2iGN5EOBKb8X_QY5USy8_7c8aaE5Dahd3nuv6g4X8EJlP7Y68mTQh7cZDrd5qh4Gd3J4EQkL5Acmeau2kK_Qn-xFLtEMaCgYKARESARISFQHGX2MiqCWM09LEuZ4lkj4xXLk0TQ0426" \
-H "Content-Type: application/json" \
https://us-central1-aiplatform.googleapis.com/v1/projects/gen-lang-client-0562904075/locations/us-central1/publishers/google/models/gemini-1.5-flash-001:generateContent -d \
$'{
  "contents": [
    {
      "role": "USER",
      "parts": { "text": "Hello!" }
    },
    {
      "role": "MODEL",
      "parts": { "text": "Argh! What brings ye to my ship?" }
    },
    {
      "role": "USER",
      "parts": { "text": "Wow! You are a real-life priate!" }
    }
  ],
}'
    """

    def __init__(self, llm) -> None:
        BaseService.__init__(self, llm)
        self.model_name = llm.split("://")[1]
        self.project_id = os.getenv("VERTEX_PROJECT_ID")

        # Use a higher rate limit for gemini 1.0
        if self.model_name == "gemini-1.0-pro":
            # https://console.cloud.google.com/iam-admin/quotas?_ga=2.158585530.1411846923.1714075021-1901817751.1714074938&pageState=(%22allQuotasTable%22:(%22s%22:%5B(%22i%22:%22displayName%22,%22s%22:%220%22),(%22i%22:%22effectiveLimit%22,%22s%22:%221%22),(%22i%22:%22currentPercent%22,%22s%22:%221%22),(%22i%22:%22sevenDayPeakPercent%22,%22s%22:%220%22),(%22i%22:%22currentUsage%22,%22s%22:%221%22),(%22i%22:%22sevenDayPeakUsage%22,%22s%22:%220%22),(%22i%22:%22serviceTitle%22,%22s%22:%220%22),(%22i%22:%22displayDimensions%22,%22s%22:%220%22)%5D,%22f%22:%22%255B%257B_22k_22_3A_22_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22base_model_3Agemini-pro_5C_22_22_2C_22s_22_3Atrue%257D%255D%22))&authuser=1&project=gen-lang-client-0562904075
            self.max_requests_per_minute = 250
        else:
            # https://console.cloud.google.com/iam-admin/quotas?_ga=2.158585530.1411846923.1714075021-1901817751.1714074938&pageState=(%22allQuotasTable%22:(%22s%22:%5B(%22i%22:%22displayName%22,%22s%22:%220%22),(%22i%22:%22effectiveLimit%22,%22s%22:%221%22),(%22i%22:%22currentPercent%22,%22s%22:%221%22),(%22i%22:%22sevenDayPeakPercent%22,%22s%22:%220%22),(%22i%22:%22currentUsage%22,%22s%22:%221%22),(%22i%22:%22sevenDayPeakUsage%22,%22s%22:%220%22),(%22i%22:%22serviceTitle%22,%22s%22:%220%22),(%22i%22:%22displayDimensions%22,%22s%22:%220%22)%5D,%22f%22:%22%255B%257B_22k_22_3A_22_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22base_model_3Agemini-1.5-pro_5C_22_22_2C_22s_22_3Atrue%257D%255D%22))&authuser=1&project=gen-lang-client-0562904075
            self.max_requests_per_minute = 55

    def __api_key(self):
        return os.getenv("VERTEX_API_KEY")

    def request_url(self):
        return f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/us-central1/publishers/google/models/{self.model_name}:generateContent"

    def request_header(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key()}",
        }

    def sample_request(self):
        return {
            "contents": [
                {"role": "USER", "parts": {"text": "Say hello!"}},
            ],
        }

    def rate_limit_time_unit(self):
        return "minutes"

    def max_requests_per_unit(self):
        return self.max_requests_per_minute

    def get_request_prompt(self, request: dict) -> str:
        return request["contents"][0]["parts"][0]["text"]

    def get_request_body(
        self, user_prompt: str, temperature: float | None, schema_name: str | None
    ) -> dict:
        if schema_name is not None:
            logging.warning(
                f"Vertex does not support structured output. Skipping schema: {schema_name}."
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
class CerebrasService(BaseService):
    """https://api.cerebras.ai/v1/chat/completions"""

    def __init__(self, llm) -> None:
        BaseService.__init__(self, llm)
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


PROVIDER_REGISTRY = {
    "vertex": VertexService,
    "mistral": MistralService,
    "together": TogetherService,
    "cohere": CohereService,
    "anthropic": AnthropicService,
    "openai": OpenAIService,
    "cerebras": CerebrasService,
}


def get_service_for_llm(llm: str):
    provider_name = get_provider_name(llm)
    return PROVIDER_REGISTRY[provider_name](llm)
