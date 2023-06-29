from enum import Enum
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.prompts import PromptTemplate


from modules.evaluator_llm import LLM
from helpers.prompt_utils import (
    GRADE_ANSWER_PROMPT,
    GRADE_ANSWER_PROMPT_BIAS_CHECK,
    GRADE_ANSWER_PROMPT_FAST,
    GRADE_ANSWER_PROMPT_OPENAI,
    GRADE_DOCS_PROMPT,
    GRADE_DOCS_PROMPT_FAST,
)

class GradeAnswerTypeEnum(Enum):
    FAST = 'Fast'
    DESCRIPTIVE_WITH_BIAS = 'Descriptive w/ bias check'
    OPEN_AI = 'OpenAI grading prompt'
    DEFAULT = 'Default'

class GradeDocsTypeEnum(Enum):
    FAST = 'Fast'
    DEFAULT = 'Default'

class GraderChain:
    __grade_answer_prompt: PromptTemplate
    __retrieval_prompt: PromptTemplate
    __grade_answer_chain: QAEvalChain
    __grade_retrieval_chain: QAEvalChain

    def __init__(self, grade_answer_type: GradeAnswerTypeEnum, grade_docs_type: GradeDocsTypeEnum):
        if grade_answer_type not in GradeAnswerTypeEnum:
            raise ValueError("Invalid Chain Type for Grading Answers")

        if grade_docs_type not in GradeDocsTypeEnum:
            raise ValueError("Invalid Chain Type for Grading Doc Retrievals")

        # Instantiate the answer evaluation chain
        answer_prompt = None
        if grade_answer_type == GradeAnswerTypeEnum.FAST:
            answer_prompt = GRADE_ANSWER_PROMPT_FAST
        elif grade_answer_type == GradeAnswerTypeEnum.DESCRIPTIVE_WITH_BIAS:
            answer_prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
        elif grade_answer_type == GradeAnswerTypeEnum.OPEN_AI:
            answer_prompt = GRADE_ANSWER_PROMPT_OPENAI
        else:
            answer_prompt = GRADE_ANSWER_PROMPT

        # Note: GPT-4 grader is advised by OAI
        self.__grade_answer_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-4",
                                                                        temperature=0), 
                                                         prompt=answer_prompt)

        # Instantiate the doc retrieval evaluation chain
        doc_prompt = None
        if grade_docs_type == GradeDocsTypeEnum.FAST:
            doc_prompt = GRADE_DOCS_PROMPT_FAST
        else:
            doc_prompt = GRADE_DOCS_PROMPT

        self.__grade_retrieval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-4",
                                                                            temperature=0), 
                                                            prompt=doc_prompt)

    def grade_answer(self, truths, generated_answers):
        return self.__grade_answer_chain.evaluate(examples=truths,
                                           predictions=generated_answers,
                                           question_key="question",
                                           prediction_key="result")

    def grade_retrieval(self, truths, generated_answers):
        return self.__grade_retrieval_chain.evaluate(examples=truths,
                                              predictions=generated_answers,
                                              question_key="question",
                                              prediction_key="result")

