from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class Status(str, Enum):
    ACTIVE = "active"
    # Still around and queryable, but not recommended for new use.
    OBSOLESCENT = "obsolescent"
    # No longer available for use, but still registered for historical purposes.
    DEPRECATED = "deprecated"


class CostConfig(BaseModel):
    per_1m_input_tokens: float
    per_1m_input_tokens_cached: Optional[float] = None
    per_1m_output_tokens: float


class Costs(BaseModel):
    serverless: Optional[CostConfig]
    batch: Optional[CostConfig] = None


class ModelProvider(BaseModel):
    # The name of the provider.
    name: str

    # Fully qualified name.
    fully_qualified_name: str

    # The status of the model.
    status: Status = Status.ACTIVE

    # The maximum number of tokens that will be output in a single request. None if there is no
    # inherent limit.
    max_output_tokens: str | int = None

    # The cost of the model under this provider.
    costs: Costs


class ModelInfo(BaseModel):
    # The maximum number of tokens that can be used in a single request, inclusive of both input and
    # output tokens.
    context_window: str | int

    # The training data cutoff date.
    training_data_cutoff: str

    # The release date of the model.
    release_date: str

    # The number of parameters in the model.
    num_parameters: str

    # List of modalities this model supports.
    modalities: List[Modality]


class LanguageModel(BaseModel):
    # The name of the model.
    model_name: str

    # The name of the provider.
    providers: list[ModelProvider]

    model_info: ModelInfo

    class Config:
        # Disable protected namespaces for "model_"
        protected_namespaces = ()

        # This will allow the use of enum values directly as strings
        use_enum_values = True
