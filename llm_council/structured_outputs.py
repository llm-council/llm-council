from pydantic import BaseModel
from typing import Annotated, Dict, Type


class BaseSchema(BaseModel):

    @staticmethod
    def method(*args, **kwargs):
        pass


class ReasoningThenAnswer(BaseSchema):
    reasoning: str
    answer: str

    @staticmethod
    def method(reasoning: Annotated[str, ""], answer: Annotated[str, ""]):
        pass


class AnswerThenReasoning(BaseSchema):
    answer: str
    reasoning: str

    @staticmethod
    def method(answer: Annotated[str, ""], reaonsing: Annotated[str, ""]):
        pass


class AnswerOnly(BaseSchema):
    answer: str

    @staticmethod
    def method(answer: Annotated[str, ""]):
        pass


class User(BaseSchema):
    name: str
    age: int


class DirectAssessmentCriteria(BaseSchema):
    name: str
    value: int


class MultipleDirectAssessmentCriteria(BaseSchema):
    criteria: Dict[str, DirectAssessmentCriteria]


STRUCTURED_OUTPUT_REGISTRY: Dict[str, Type[BaseSchema]] = {
    "reasoning_then_answer": ReasoningThenAnswer,
    "answer_then_reasoning": AnswerThenReasoning,
    "answer_only": AnswerOnly,
    "test_user": User,
    "direct_assessment": MultipleDirectAssessmentCriteria,
}
