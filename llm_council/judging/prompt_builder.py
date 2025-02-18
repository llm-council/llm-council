# from llm_council.judging.schema import DirectAssessmentConfig


DIRECT_ASSESSMENT_JUDGING_BASE_TEMPLATE = """We would like to evaluate the quality of the response.

### USER PROMPT START ###
{user_prompt}
### USER PROMPT END ###

### RESPONSE START ###
{response}
### RESPONSE END ###

Please evaluate the quality of the response based on the following criteria:

{criteria_verbalized}

Options:

{likert_scale_verbalized}
"""

# Please return your rating as a JSON object with the key as the criteria and the rating as an integer value. for example:

# {{
#     <criteria>: <rating>,
#     "Relevance": <rating>,
# }}


# Placeholders that are handled by the prompt builder.
DIRECT_ASSESSMENT_SPECIAL_PLACEHOLDERS = [
    "criteria_verbalized",
    "likert_scale_verbalized",
    # "sample_return_object",
]


PREBUILT_LIKERT_SCALE_2 = """Options:
- 1: disagree
- 2: agree
"""

PREBUILT_LIKERT_SCALE_3 = """Options:
- 1: disagree
- 2: neither agree nor disagree
- 3: agree
"""

PREBUILT_LIKERT_SCALE_4 = """Options:
- 1: strongly disagree
- 2: disagree
- 3: agree
- 4: strongly agree
"""

PREBUILT_LIKERT_SCALE_5 = """Options:
- 1: strongly disagree
- 2: disagree
- 3: neither agree nor disagree
- 4: agree
- 5: strongly agree
"""

PREBUILT_LIKERT_SCALE_6 = """Options:
- 1: strongly disagree
- 2: disagree
- 3: slightly disagree
- 4: slightly agree
- 5: agree
- 6: strongly agree
"""

PREBUILT_LIKERT_SCALE_7 = """Options:
- 1: strongly disagree
- 2: disagree
- 3: slightly disagree
- 4: neither agree nor disagree
- 5: slightly agree
- 6: agree
- 7: strongly agree
"""


LIKERT_PREBUILT_MAP = {
    2: PREBUILT_LIKERT_SCALE_2,
    3: PREBUILT_LIKERT_SCALE_3,
    4: PREBUILT_LIKERT_SCALE_4,
    5: PREBUILT_LIKERT_SCALE_5,
    6: PREBUILT_LIKERT_SCALE_6,
    7: PREBUILT_LIKERT_SCALE_7,
}


# def generate_criteria_verablized(direct_assessment_config: DirectAssessmentConfig):
#     criteria_verbalized = ""
#     for i, criterion in enumerate(direct_assessment_config.criteria, start=1):
#         criteria_verbalized += (
#             f"{i}. {criterion.name}: {criterion.criteria_statement}\n"
#         )
#     return criteria_verbalized


# def generate_likert_scale_verbalized(likert_scale):
#     return LIKERT_PREBUILT_MAP[likert_scale]


# import jsonschema
# import hypothesis_jsonschema
# import json

# # Example JSON schema
# schema = {
#     "type": "object",
#     "properties": {
#         "name": {"type": "string"},
#         "age": {"type": "integer", "minimum": 18},
#         "email": {"type": "string", "format": "email"},
#         "is_active": {"type": "boolean"},
#         "preferences": {"type": "array", "items": {"type": "string"}},
#     },
#     "required": ["name", "age", "email", "is_active"],
# }

# # Generate a sample object
# sample_object = hypothesis_jsonschema.from_schema(schema).example()
