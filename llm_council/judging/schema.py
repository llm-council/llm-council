from typing import List, Literal, Union
from pydantic import BaseModel, Field, create_model
from typing import Type
from llm_council.judging.prompt_builder import DIRECT_ASSESSMENT_JUDGING_BASE_TEMPLATE
import json


class DirectAssessmentCriteria(BaseModel):
    """Defines a single evaluation criterion for direct assessment."""

    name: str = Field(..., description="The name of the evaluation criterion.")
    criteria_statement: str = Field(
        ..., description="A detailed description of what is being assessed."
    )


class DirectAssessmentConfig(BaseModel):
    """Configuration schema for direct assessment evaluations."""

    prompt_template: str = Field(
        ..., description="The prompt template used to guide the LLM’s assessment."
    )
    criteria: List[DirectAssessmentCriteria] = Field(
        ..., description="A list of criteria that define the evaluation dimensions."
    )
    prebuilt_likert_scale: Literal[2, 3, 4, 5, 6, 7] = Field(
        ...,
        description="The Likert scale size used for scoring (e.g., 2, 3, 4, 5, 6, or 7).",
    )


class PairwiseComparisonConfig(BaseModel):
    """Configuration schema for pairwise comparison evaluations."""

    prompt_template: str = Field(
        ..., description="The prompt template used for pairwise comparison."
    )
    themes_to_consider: List[str] = Field(
        ...,
        description="A list of themes or aspects that should be considered during comparison.",
    )
    granularity: Literal[2, 3, 4, 5] = Field(
        ...,
        description="The level of granularity for ranking (e.g., binary win/loss vs. finer scales).",
    )
    reference_llm: str = Field(
        ...,
        description="The reference LLM that the other models are compared against.",
    )


class EvaluationConfig(BaseModel):
    """Defines the overall schema for an LLM-based evaluation framework."""

    type: Literal["direct_assessment", "pairwise_comparison"] = Field(
        ...,
        description="Specifies whether the evaluation is direct assessment or pairwise comparison.",
    )
    exclude_self_grading: bool = Field(
        ...,
        description="If True, the system ensures that models do not assess their own outputs.",
    )
    cot_enabled: bool = Field(
        False,
        description="If True, the LLM is prompted to provide a chain-of-thought reasoning before giving its final judgment.",
    )
    config: Union[DirectAssessmentConfig, PairwiseComparisonConfig] = Field(
        ...,
        description="The configuration object, which depends on the evaluation type.",
    )


def create_dynamic_schema(eval_config: EvaluationConfig) -> Type[BaseModel]:
    """Dynamically creates a Pydantic schema class based on the EvaluationConfig input."""
    if eval_config.type != "direct_assessment":
        raise ValueError("Currently only supports direct assessment.")

    # Dynamically define fields based on provided criteria
    fields = {
        criterion.name: (int, Field(..., description=criterion.criteria_statement))
        for criterion in eval_config.config.criteria
    }

    # Use Pydantic's create_model to generate a new schema class dynamically
    schema_class = create_model("DynamicAssessmentSchema", **fields)

    return schema_class


def save_config(config: EvaluationConfig, file_path: str):
    """Saves a Pydantic EvaluationConfig object to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(config.dict(), f, indent=4)


def load_config(file_path: str) -> EvaluationConfig:
    """Loads an EvaluationConfig object from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return EvaluationConfig.parse_obj(data)


# Basic evaluation config for direct assessment.
DEFAULT_EVALUATION_CONFIG = EvaluationConfig(
    type="direct_assessment",
    exclude_self_grading=True,
    cot_enabled=True,
    config=DirectAssessmentConfig(
        prompt_template=DIRECT_ASSESSMENT_JUDGING_BASE_TEMPLATE,
        prebuilt_likert_scale=5,
        criteria=[
            DirectAssessmentCriteria(
                name="Coherence",
                criteria_statement="The response is coherent.",
            ),
            DirectAssessmentCriteria(
                name="Relevance",
                criteria_statement="The response is relevant.",
            ),
        ],
    ),
)
