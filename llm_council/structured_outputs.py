from pydantic import BaseModel


class ReasoningThenAnswer(BaseModel):
    reasoning: str
    answer: str


class AnswerThenReasoning(BaseModel):
    answer: str
    reasoning: str


class AnswerOnly(BaseModel):
    answer: str


STRUCTURED_OUTPUT_REGISTRY = {
    "reasoning_then_answer": ReasoningThenAnswer,
    "answer_then_reasoning": AnswerThenReasoning,
    "answer_only": AnswerOnly,
}
