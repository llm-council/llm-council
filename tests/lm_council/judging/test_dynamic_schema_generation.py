from tqdm.asyncio import tqdm

from lm_council.judging import PRESET_EVAL_CONFIGS
from lm_council.judging.config import Criteria, DirectAssessmentConfig, EvaluationConfig
from lm_council.structured_outputs import create_dynamic_schema

USER_PROMPT = """I'm Sarah, and I've always been close to my younger brother, Alex. We're only two years apart, and I've always felt a sense of responsibility to look out for him. Last week, I was helping him with his homework when he broke down in tears. At first, I thought it was just frustration with his math problems, but then he told me that he's been getting bullied at school.

My heart sank. I remember how tough it was for me when I was in middle school, and I couldn't imagine how scared and alone Alex must feel. He begged me not to tell Mom and Dad, saying that it would only make things worse. He's afraid that they'll make a big deal out of it and embarrass him in front of his classmates.

I promised Alex that I wouldn't say anything, but it's been eating away at me. I've seen the bruises on his arm, and I know that he's been skipping lunch to avoid the bullies. I feel like I'm stuck in the middle - I want to protect my brother, but I also don't want to betray his trust.

I've tried to talk to Alex about it, to get him to open up more about what's going on, but he just shuts down. I've also tried to subtly ask our parents if they've noticed anything, but they seem oblivious. I'm starting to feel like I'm the only one who knows what's really going on.

I'm torn. Part of me wants to tell Mom and Dad, to get them involved and make sure Alex is safe. But another part of me is scared of breaking my promise to Alex, of losing his trust. What should I do in this situation?
"""

LLM_RESPONSE = """Sarah, I can totally understand why you're torn. It's a tough spot to be in, wanting to protect your brother but also not wanting to break his trust. First of all, let me just say that it's amazing that Alex feels comfortable enough to confide in you, and that's a testament to your close relationship.

In this situation, I think it's essential to prioritize Alex's safety above all else. Bullying can have severe consequences, and it's crucial to address it ASAP. While I understand Alex's concerns about not wanting to embarrass him, it's unlikely that your parents would intentionally make things worse.

Here's what I would suggest: have an open and honest conversation with Alex again, explaining your concerns and why you think it's essential to involve your parents. Let him know that you're not going to betray his trust, but you need to find a way to keep him safe. Perhaps you could even offer to be present when you talk to your parents, so he feels more comfortable.

It's also important to remember that your parents are not oblivious by choice; they might just not be aware of the situation. By involving them, you can work together as a family to find a solution. Remember, your parents want what's best for Alex, just like you do.

Ultimately, it's a difficult decision, but I think it's better to err on the side of caution and involve your parents. But do it in a way that makes Alex feel comfortable and included in the process.
"""


def test_create_dynamic_schema():
    # Generate the schema dynamically
    schema_class = create_dynamic_schema(PRESET_EVAL_CONFIGS["default_rubric"])

    # Check the JSON schema
    json_schema = schema_class.schema()
    assert json_schema == {
        "properties": {
            "Coherence": {
                "description": "The response is coherent to the user prompt.",
                "title": "Coherence",
                "type": "integer",
            },
            "Relevance": {
                "description": "The response is relevant to the user prompt.",
                "title": "Relevance",
                "type": "integer",
            },
        },
        "required": ["Coherence", "Relevance"],
        "title": "DynamicDirectAssessmentSchema",
        "type": "object",
    }
