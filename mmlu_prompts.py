STRUCTURED_OUTPUT_JUDGE_GROUND_TRUTH_COT_FIRST = '''The following multiple-choice question was asked to a {role}.

Question: {question}

Options: {options}

The {role}'s answer was:
{answer}

Please assess whether the {role} is correct or incorrect, or if you do not know. 

Think step by step and then decide on your final answer. Your final answer should be a letter choice that corresponds to:
A: the {role} is correct
B: the {role} is wrong
C: I am not sure or I am unable to assess the correctness confidently.
'''

STRUCTURED_OUTPUT_JUDGE_GROUND_TRUTH_COT_SECOND = '''The following multiple-choice question was asked to a {role}.

Question: {question}

Options: {options}

The {role}'s answer was:
{answer}

Please assess whether the {role} is correct or incorrect, or if you do not know. 

Provide your final answer, and then provide your reasoning for it. Your final answer should be a letter choice that corresponds to:
A: the {role} is correct
B: the {role} is wrong
C: I am not sure or I am unable to assess the correctness confidently.
'''

STRUCTURED_OUTPUT_JUDGE_GROUND_TRUTH_NO_COT = '''The following multiple-choice question was asked to a {role}.

Question: {question}

Options: {options}

The {role}'s answer was:
{answer}

Please assess whether the {role} is correct or incorrect, or if you do not know. 

Your final answer should be a letter choice that corresponds to:
A: the {role} is correct
B: the {role} is wrong
C: I am not sure or I am unable to assess the correctness confidently.
'''

PROMPT_JUDGE_GROUND_TRUTH_COT_FIRST = """The following multiple-choice question was asked to a {role}.

Question: {question}

Options: {options}

The {role}'s answer was:
{answer}

Please assess whether the {role} is correct or incorrect, or if you do not know. 

Think step by step and then finish your answer with "My assessment is (X)." where X is the letter choice that corresponds to:
A: the {role} is correct
B: the {role} is wrong
C: I am not sure or I am unable to assess the correctness confidently.
"""

PROMPT_JUDGE_GROUND_TRUTH_COT_SECOND = """The following multiple-choice question was asked to a {role}.

Question: {question}

Options: {options}

The {role}'s answer was:
{answer}

Please assess whether the {role} is correct or incorrect, or if you do not know. 

Begin your answer with "My assessment is (X)." where X is the letter choice that corresponds to:
A: the {role} is correct
B: the {role} is wrong
C: I am not sure or I am unable to assess the correctness confidently.

Then, provide your reasoning for your answer.
"""

PROMPT_JUDGE_GROUND_TRUTH_NO_COT = """The following multiple-choice question was asked to a {role}.

Question: {question}

Options: {options}

The {role}'s answer was:
{answer}

Please assess whether the {role} is correct or incorrect, or if you do not know. 

Begin your answer with "My assessment is (X)." where X is the letter choice that corresponds to:
A: the {role} is correct
B: the {role} is wrong
C: I am not sure or I am unable to assess the correctness confidently.
"""


PROMPT_ANSWER_COT_FIRST = '''The following are multiple-choice questions (with answers). Think step by step and then finish your answer with "The answer is (X)" where X is the correct letter choice.

Question: {question}

Options: {options}
'''


PROMPT_ANSWER_COT_SECOND = '''The following are multiple-choice questions (with answers). Respond with: "The answer is (X)" where X is the correct letter choice. Then, provide your reasoning for it.

Question: {question}

Options: {options}
'''

STRUCTURED_OUTPUT_ANSWER_COT_FIRST = '''The following are multiple-choice questions (with answers). Think step by step, and then respond with a JSON payload with your reasoning and your final answer corresponding to the correct letter choice.

Question: {question}

Options: {options}
'''


STRUCTURED_OUTPUT_ANSWER_COT_SECOND = '''The following are multiple-choice questions (with answers). Respond with a JSON payload with your final answer corresponding to the correct letter choice and then your reasoning for it.

Question: {question}

Options: {options}
'''

STRUCTURED_OUTPUT_ANSWER_NO_COT = '''The following are multiple-choice questions (with answers). Respond with a JSON payload with your final answer corresponding to the correct letter choice.

Question: {question}

Options: {options}
'''

PROMPT_ANSWER_NO_COT = '''The following are multiple-choice questions (with answers). Begin your response with "The answer is (X)" where X is the correct letter choice.

Question: {question}

Options: {options}
'''
