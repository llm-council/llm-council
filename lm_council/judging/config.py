import json
from typing import List, Literal, Type, Union

from pydantic import BaseModel, Field, create_model

from lm_council.judging.prompt_builder import DIRECT_ASSESSMENT_JUDGING_BASE_TEMPLATE


class Criteria(BaseModel):
    """Defines a single evaluation criterion for direct assessment."""

    name: str = Field(..., description="The name of the evaluation criterion.")
    statement: str = Field(
        ..., description="A detailed description of what is being assessed."
    )


class DirectAssessmentConfig(BaseModel):
    """Configuration schema for direct assessment evaluations."""

    prompt_template: str = Field(
        ..., description="The prompt template used to guide the LLM’s assessment."
    )
    rubric: List[Criteria] = Field(
        ..., description="A list of criteria that define the evaluation dimensions."
    )
    prebuilt_likert_scale: Literal[2, 3, 4, 5, 6, 7] = Field(
        ...,
        description="The Likert scale size used for scoring (e.g., 2, 3, 4, 5, 6, or 7).",
    )


class PairwiseComparisonFixedReferencesConfig(BaseModel):
    reference_models: List[str] = Field(
        ...,
        description="The reference LLMs that the other models are compared against.",
    )


class PairwiseComparisonRandomMatchesConfig(BaseModel):
    n_random_pairs: int = Field(
        ...,
        description="The number of random pairs to generate for each comparison.",
    )


class PairwiseComparisonConfig(BaseModel):
    """Configuration schema for pairwise comparison evaluations."""

    prompt_template: str = Field(
        ..., description="The prompt template used for pairwise comparison."
    )
    granularity: Literal[2, 3, 4, 5] = Field(
        ...,
        description="""
The level of granularity for ranking (e.g., binary win/loss vs. finer scales).

2: Binary (A wins, B wins)
3: Ternary (A wins, tie, B wins)
4: Quaternary (A wins, A slightly wins, B slightly wins, B wins)
5: Quinary (A wins, A slightly wins, tie, B slightly wins, B wins)
""",
    )
    skip_equal_pairs: bool = Field(
        True,
        description="If True, the system will skip pairs where the two models are the same.",
    )
    algorithm_type: Literal["fixed_reference_models", "random", "all_pairs"] = Field(
        ...,
        description="The algorithm used to generate pairwise comparisons.",
    )
    position_flipping: bool = Field(
        True,
        description="If True, exhaustively flip the position of the pairwise comparisons being generated.",
    )
    algorithm_config: Union[
        PairwiseComparisonFixedReferencesConfig,
        PairwiseComparisonRandomMatchesConfig,
        None,
    ] = Field(
        None,
        description="The configuration for the pairwise comparison algorithm. Can be None.",
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
    temperature: float = Field(
        0.0,
        description="The temperature setting for the LLM judge, controlling the randomness of the output.",
    )
    config: Union[DirectAssessmentConfig, PairwiseComparisonConfig] = Field(
        ...,
        description="The configuration object, which depends on the evaluation type.",
    )
    reps: int = Field(
        1,
        description="The number of repetitions for each evaluation instance.",
    )

    def save_config(self, file_path: str):
        """Saves a Pydantic EvaluationConfig object to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def load_config(cls, file_path: str) -> "EvaluationConfig":
        """Loads an EvaluationConfig object from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.parse_obj(data)
