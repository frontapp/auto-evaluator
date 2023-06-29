from enum import Enum
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA


from modules.evaluator_llm import LLM
from helpers.prompt_utils import QA_CHAIN_PROMPT, REFINE_QA_CHAIN_PROMPT
from modules.retriever import EvaluatorRetrieverType

class QuestionAnswerChainEnum(Enum):
    STUFF = 'stuff'
    REFINE = 'refine'
    MAP_REDUCE = 'map_reduce'
    MAP_RERANK = 'map_rerank'

class QuestionAnswerChain:
    __chain: BaseRetrievalQA

    def __init__(self, llm: LLM, retriever: EvaluatorRetrieverType, chain_type: QuestionAnswerChainEnum):
        if chain_type not in QuestionAnswerChainEnum:
            raise ValueError("Invalid Chain Type for Q/A")


        # Figure out which prompts to use
        chain_type_kwargs = {}
        if chain_type == QuestionAnswerChainEnum.STUFF:
            chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
        elif chain_type == QuestionAnswerChainEnum.REFINE:
            chain_type_kwargs = {"question_prompt": QA_CHAIN_PROMPT,
                                 "refine_prompt": REFINE_QA_CHAIN_PROMPT}

        # Load the chain
        self.__chain = RetrievalQA.from_chain_type(llm,
                                                   chain_type=chain_type.value,
                                                   retriever=retriever,
                                                   chain_type_kwargs=chain_type_kwargs,
                                                   return_source_documents=True)

    def call_chain(self, query: str):
        return self.__chain({"query": query})
