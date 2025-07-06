from typing import Annotated, Dict, Literal, Type, get_args, get_origin

from pydantic import BaseModel, Field, create_model

PAIRWISE_COMPARISON_LABEL_MAP = {
    2: """[[A>B]]: The first response better.
[[B>A]]: The second response is better.
""",
    3: """[[A>B]]: The first response is better.
[[A=B]]: The two responses are equally good.
[[B>A]]: The second response is better.
    """,
    4: """[[A>>B]]: The first response is significantly better.
[[A>B]]: The first response is slightly better.
[[B>A]]: The second response is slightly better.
[[B>>A]]: The second response is significantly better.
""",
    5: """[[A>>B]]: The first response is significantly better.
[[A>B]]: The first response is slightly better.
[[A=B]]: The two responses are equally good.
[[B>A]]: The second response is slightly better.
[[B>>A]]: The second response is significantly better.
""",
}


def _labels_for(granularity: int) -> list[str]:
    """Extract the raw bracket labels (e.g. '[[A>B]]') for a given granularity."""
    if granularity not in PAIRWISE_COMPARISON_LABEL_MAP:
        raise ValueError(
            f"Granularity must be one of {tuple(PAIRWISE_COMPARISON_LABEL_MAP)}"
        )
    # keep only the token inside the leading dash & space
    lines = PAIRWISE_COMPARISON_LABEL_MAP[granularity].strip().splitlines()
    return [line.split(":")[0].strip() for line in lines]  # e.g. '[[A>B]]'


def get_pairwise_comparison_schema(
    granularity: int, cot_enabled: bool
) -> type[BaseModel]:
    """
    Dynamically create and return a pydantic BaseModel that has:
      • pairwise_choice   – limited to the allowed bracket codes for the granularity
      • explanation – optional free‑text field (first, if `cot_enabled`)
    """
    labels = _labels_for(granularity)

    # Build a Literal type such as Literal['[[A>B]]', '[[B>A]]', ...]
    RatingType = Literal[tuple(labels)]  # type: ignore[arg-type]

    # Field definitions in the shape expected by `create_model`
    fields: dict[str, tuple[type, Field]] = {
        "pairwise_choice": (
            RatingType,
            Field(..., description=f"One of: {', '.join(labels)}"),
        )
    }

    if cot_enabled:
        # Insert 'explanation' before 'pairwise_choice' to force thinking before the answer.
        fields = {
            "explanation": (
                str,
                Field(..., description="Free‑text chain‑of‑thought reasoning"),
            ),
            **fields,
        }

    # create_model(name, **field_definitions)
    return create_model(
        f"PairwiseComparisonG{granularity}{'_COT' if cot_enabled else ''}",
        **fields,
    )


def create_dynamic_schema(eval_config) -> Type[BaseModel]:
    """Dynamically creates a Pydantic schema class based on the EvaluationConfig input."""
    if eval_config.type != "direct_assessment":
        raise ValueError("Currently only supports direct assessment.")

    # Dynamically define fields based on provided criteria
    fields = {
        criterion.name: (int, Field(..., description=criterion.statement))
        for criterion in eval_config.config.rubric
    }

    # Use Pydantic's create_model to generate a new schema class dynamically
    schema_class = create_model("DynamicDirectAssessmentSchema", **fields)

    return schema_class
