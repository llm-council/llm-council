# Taken from Arena Hard: https://github.com/lm-sys/arena-hard-auto/blob/main/config/judge_config.yaml
JUDGE_CHATBOT_ARENA_EQ_SUBSET_COMPLETIONS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:
- [[A>>B]]: The first response is significantly better.
- [[A>B]]: The first response is slightly better.
- [[B>A]]: The second response is slightly better.
- [[B>>A]]: The second response is significantly better.

### USER PROMPT START ###
{context}
### USER PROMPT END ###

### ASSISTANT A RESPONSE START ###
{first_completion}
### ASSISTANT A RESPONSE END ###

### ASSISTANT B RESPONSE START ###
{second_completion}
### ASSISTANT B RESPONSE END ###
"""


ALPACA_EVAL_EVALUATION_TEMPLATE = """<|im_start|>system
You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers.
<|im_end|>
<|im_start|>user
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{
    "instruction": "{instruction}",
}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{
    {
        "model_identifier": "m",
        "output": "{output_1}"
    },
    {
        "model_identifier": "M",
        "output": "{output_2}"
    }
}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by first providing a concise explanation and then end your answer by providing the model identifier of the best output. We will use the last character of your output `output[-1]` as the name of the best model, so make sure you finish with the token of the model identifiers and nothing else: `m` or `M` (no quotes, no dots, no backticks, no new lines, ...). For example:

### Concise explanation
...some text...

### Which is best, m or M?
M

Now is your turn.

## Your answer: "Concise explanation" followed by "Which is best, m or M?"
<|im_end|>"""

GENERATE_EXPANSION = """This an example of an interesting interpersonal conflict.

"I was truthful when admitting to my wife that I found her unattractive after her surgery.

My wife had plastic surgery recently. We had discussed it and I was against it. It was not my decision and ultimately I had no say.

She looks weird now. She had the fat sucked out of her face, lip fillers, a neck lift, other stuff I don't really get.

She gives me uncanny valley vibes now. It freaks me out. She is fully healed now and she wants us to go back to normal. Like me initiating sex. I have done so but not as much as I used to. And when I do I try and make sure there is very little light.

It's been a few months and I kind of dread having to look at her. Obviously she has noticed. She has been bugging me to tell her what's up. I've tried telling her I'm just tired from work. Or that I'm run down. Really anything except for the truth.

She broke down and asked me if I was having an affair. I said that I wasn't. She asked to look at my phone. I unlocked it for her and handed it over. I wasn't worried about her finding anything because there is nothing to find. She spent an hour looking through it and found nothing. She asked me to explain why I changed. I tried explaining that I just wasn't that interested right now.

Nothing I said was good enough for her. She kept digging. I finally told the truth. I wasn't harsh or brutally honest. I just told her that her new face wasn't something I found attractive and that I was turned off. She asked if that's why I turn off all the lights now. I said yes. She started crying and said that she needed time alone. She went to stay with her sister.

I have been called every name in the book since this happened. Her sister said I'm a piece of shit for insulting my wife's looks. Her friends all think I'm the asshole.

I tried not to say anything. I can't force myself to find her attractive. I still love her but her face is just weird now. She looks like the blue alien from The Fifth Element.

What should I do in this situation?"

Now we would like you to expand upon an interpersonal conflict for a different hypothetical person:

"{scenario_text}"

Please write the story in first person. The story should be 250 to 350 words in length. Provide adequate detail, but be concise and matter-of-fact. Do not be overly verbose. Conclude with a question: "What should I do in this situation?
"""

ASSESS_VALUE_IN_EQ_TEST = """We aim to create a test set of emotionally nuanced scenarios to be used in an emotional intelligence test where the task is to respond to an emotionally sophisticated scenario.

Here is a scenario that we are considering including in the test:

"{response_string}"

Begin your evaluation by providing a short 1-sentence summary of the scenario described. Some themes to consider in your evaluation: realism, relatability, nuance, emotional depth, uniqueness and creativity, verbosity, and clarity.

After providing your explanation, please tell us how much you agree with the following statement:

Statement: This scenario represents an emotionally compelling situation that would be excellent to use in an emotional intelligence test where the task is to dialectically respond to this situation.

Please provide a rating as one of the following options:
- Strongly disagree
- Disagree
- Slightly disagree
- Neither agree nor disagree
- Slightly agree
- Agree
- Strongly agree
"""

CONVERSATIONAL_DILEMMA_RESPONSE = """The person here is seeking advice:

"{response_string}"

How would you respond to this person in a conversation?

Please limit your response to {word_limit} words at most. Any response exceeding this limit will be truncated.
"""

LIKERT_AITA_ASSESSMENT = """The person here is seeking advice:

### CONTEXT START ###
{context}
### CONTEXT END ###

Provide a short explanation describing how you would evaluate the post for the following statement:

Statement: "This person is being an asshole in this situation."

Then, provide your final rating for that statement. Options:
- Strongly Disagree
- Disagree
- Slightly Disagree
- Neither Agree Nor Disagree
- Slightly Agree
- Agree
- Strongly Agree
"""


JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_LIKERT = """The person here is seeking advice:

### CONTEXT START ###
{context}
### CONTEXT END ###

Someone was asked what they would actually say to this person in a conversation.

Here is how they responded:

### RESPONSE START ###
{response_string}
### RESPONSE END ###

Please evaluate the quality of the response based on the following criteria:

1. Authenticity: The response feels sincere and heartfelt, with a profound sense of personal investment and empathy.
2. Clarity: The response is lucid and easily comprehensible.
3. Empathy: The response displays a strong level of empathy and deep understanding.
4. Actionability: The response provides practical and personalized advice.
5. Verbosity: The response is succinct and to the point, avoiding unnecessary verbosity and maintaining focus.

Options:
- Strongly Disagree
- Disagree
- Slightly Disagree
- Neither Agree Nor Disagree
- Slightly Agree
- Agree
- Strongly Agree

For each of the criteria, provide a short explanation describing how you would evaluate the response based on that criterion. Then, provide your final rating for that criterion (Strongly Disagree, Disagree, Slightly Disagree, Neither Agree Nor Disagree, Slightly Agree, Agree, Strongly Agree).
"""

FORMAT_JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_LIKERT = """The following is a textual response that should describe explanations and ratings for the following categories:

- Authenticity
- Clarity
- Empathy
- Actionability
- Verbosity

The possible rating options for each criteria are:
- Strongly Disagree
- Disagree
- Slightly Disagree
- Neither Agree Nor Disagree
- Slightly Agree
- Agree
- Strongly Agree

Here is the textual response:

### RESPONSE START ###
{response_string}
### RESPONSE END ###

We want to make sure that the free-form textual response is parseable by downstream systems.

Please return a JSON object with the following keys and structure:

authenticty:
    justification: <justification>
    rating: <rating>
clarity:
    justification: <justification>
    rating: <rating>
empathy:
    justification: <justification>
    rating: <rating>
actionability:
    justification: <justification>
    rating: <rating>
verbosity:
    justification: <justification>
    rating: <rating>
"""


JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_SXS = """Someone is seeking advice from two different people.

### CONTEXT START ###
{context}
### CONTEXT END ###

### The first person's response START ###
{first_completion}
### The first person's response END ###

### The second person's response START ###
{second_completion}
### The second person's response END ###

Begin your evaluation by comparing the two responses and provide a short explanation. Some themes to consider in your evaluation: authenticity, clarity, helpfulness, empathy, actionability, verbosity.

After providing your explanation, output your final verdict as one of the following options:
- "[[A>B]]" if the first person's response is better
- "[[B<A]]" if the second person's response is better
- "[[A=B]]" if both responses are equally good
"""


JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_SXS_NO_TIE = """Someone is seeking advice from two different people.

### CONTEXT START ###
{context}
### CONTEXT END ###

### The first person's response START ###
{first_completion}
### The first person's response END ###

### The second person's response START ###
{second_completion}
### The second person's response END ###

Begin your evaluation by comparing the two responses and provide a short explanation. Some themes to consider in your evaluation: authenticity, clarity, empathy, actionability, verbosity.

After providing your explanation, output your final verdict as one of the following options:
- "[[A>B]]" if the first person's response is better
- "[[B>A]]" if the second person's response is better
"""

JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_SXS_GRANULAR_NO_TIE = """This person is seeking guidance and help regarding their emotional dilemma.

### CONTEXT START ###
{context}
### CONTEXT END ###

### The first person's response START ###
{first_completion}
### The first person's response END ###

### The second person's response START ###
{second_completion}
### The second person's response END ###

Begin your evaluation by comparing the two responses and provide a short explanation. Some themes to consider in your evaluation of the quality of responses: authenticity, clarity, empathy, actionability, verbosity.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:
- [[A>>B]]: The first response is significantly better.
- [[A>B]]: The first response is slightly better.
- [[B>A]]: The second response is slightly better.
- [[B>>A]]: The second response is significantly better.
"""

JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_SXS_GRANULAR_WITH_TIE_OPTION = """Someone is seeking advice from two different people.

### CONTEXT START ###
{context}
### CONTEXT END ###

### The first person's response START ###
{first_completion}
### The first person's response END ###

### The second person's response START ###
{second_completion}
### The second person's response END ###

Begin your evaluation by comparing the two responses and provide a short explanation. Some themes to consider in your evaluation: authenticity, clarity, empathy, actionability, verbosity.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:
- [[A>>B]]: The first response is significantly better.
- [[A>B]]: The first response is slightly better.
- [[A=B]]: The two responses are equally good.
- [[B>A]]: The second response is slightly better.
- [[B>>A]]: The second response is significantly better.
"""

QUALITATIVE_REASONING_EXTRACTION_TEMPLATE = """We would like to better qualitatively understand the reason or reasons behind the vote cast by someone who was choosing between A and B.

### VOTE START
{judging_response_string}
### VOTE END

Using the JSON indicator variable structure below as a template, please set the value to 1 for any keys that you determine is part of the basis for why this person made their preferred choice.

{{
 "more structured": 0,
 "better structured": 0,
 "more completeness": 0,
 "more succinct": 0,
 "more direct": 0,
 "more actionable": 0,
 "more deep": 0,
 "more empathetic": 0,
 "more nuanced": 0,
 "more focused": 0,
 "more accessible": 0,
 "more encouraging": 0,
 "more detailed": 0,
 "more clear": 0,
 "more conversational": 0,
 "more understanding": 0,
 "more verbose": 0,
 "less verbose": 0,
 "more personal": 0,
 "more balanced": 0,
 "more soft": 0,
 "more concrete": 0,
 "more suggestions, options, or ideas": 0,
 "better suggestions, options, or ideas": 0,
 "more digestible": 0,
 "more educational": 0,
 "more thoughtful": 0,
 "more effective": 0,
 "easier to follow": 0,
 "more specific": 0,
 "more comprehensive": 0,
 "more adaptable": 0,
 "more genuine": 0,
 "more practical": 0,
 "more gentle": 0,
 "more insightful": 0,
 "more authentic": 0,
 "other reason not listed": 0
}}

In your response, please return ONLY the JSON payload.
"""


PROMPT_REGISTRY = {
    # Test set formulation.
    "generate_expansion": GENERATE_EXPANSION,
    # Dilemma Judging.
    "assess_value_in_eq_test": ASSESS_VALUE_IN_EQ_TEST,
    # Dilemma Response.
    "conversational_dilemma_response": CONVERSATIONAL_DILEMMA_RESPONSE,
    # Dilemma Response Judging.
    "judge_conversational_dilemma_response_sxs": JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_SXS,
    "judge_conversational_dilemma_response_sxs_no_tie": JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_SXS_NO_TIE,
    "judge_conversational_dilemma_response_sxs_granular_no_tie": JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_SXS_GRANULAR_NO_TIE,
    "judge_conversational_dilemma_responses_sxs_granular_with_tie": JUDGE_CONVERSATIONAL_DILEMMA_RESPONSE_SXS_GRANULAR_WITH_TIE_OPTION,
    # Qualitative Reasoning Extraction.
    "qualitative_reasoning_extraction": QUALITATIVE_REASONING_EXTRACTION_TEMPLATE,
    # Alpaca Eval.
    "alpaca_eval_evaluation": ALPACA_EVAL_EVALUATION_TEMPLATE,
    # Chatbot arena EQ subset judging.
    "judge_chatbot_arena_eq_subset_completions": JUDGE_CHATBOT_ARENA_EQ_SUBSET_COMPLETIONS,
}


def get_realized_prompt(prompt_template_key: str, **kwargs) -> str:
    return PROMPT_REGISTRY[prompt_template_key].format(**kwargs)
