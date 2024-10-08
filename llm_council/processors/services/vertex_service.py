import dotenv
import logging
import os

from llm_council.processors.services.base_service import BaseService

dotenv.load_dotenv()


class VertexService(BaseService):
    """https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini?_gl=1*uk6ij0*_ga*MTkwMTgxNzc1MS4xNzE0MDc0OTM4*_ga_WH2QY8WWF5*MTcxNTg0MjkwNS4zOS4xLjE3MTU4NDI5MjguMC4wLjA.&_ga=2.185445036.480671911.1715842905-1901817751.1714074938
    
    Sample curl command.

curl \
-X POST \
-H "Authorization: Bearer $VERTEX_API_KEY" \
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
