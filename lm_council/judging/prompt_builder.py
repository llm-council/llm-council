import re

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


def get_placeholders(s):
    # formatter = string.Formatter()
    # return [field_name for field_name, *_ in formatter.parse(s) if field_name]
    return re.findall(r"\{([^}]+)\}", s)


def check_prompt_template_contains_all_placeholders(prompt_template, prompt_fields):
    prompt_placeholders = get_placeholders(prompt_template)
    metadata_keys = list(prompt_fields.keys()) + DIRECT_ASSESSMENT_SPECIAL_PLACEHOLDERS
    if not set(prompt_placeholders).issubset(metadata_keys):
        raise ValueError(
            f"Placeholders not accounted for: {set(prompt_placeholders) - set(metadata_keys)}."
        )
