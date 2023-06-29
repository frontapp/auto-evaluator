from enum import Enum
from typing import Union

from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic, MosaicML, Replicate

LLM = Union[ChatOpenAI, Anthropic, Replicate, MosaicML]

# The algorithm to use for splitting texts
class EvaluatorLLMEnum(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    ANTHROPIC = "anthropic"
    ANTHROPIC_100K = "Anthropic-100k"
    VICUNA_13B = "vicuna-13b"
    MOSAIC = "mosaic"

class EvaluatorLLM:
    def __init__(self, llm_type: EvaluatorLLMEnum):
        if llm_type not in EvaluatorLLMEnum:
            raise ValueError("Invalid LLM type")

        if llm_type in (llm_type.GPT_3_5_TURBO, llm_type.GPT_4):
            self.__model = ChatOpenAI(model_name=llm_type.value, temperature=0)
        elif llm_type == llm_type.ANTHROPIC:
            self.__model = Anthropic(temperature=0)
        elif llm_type == llm_type.ANTHROPIC_100K:
            self.__model = Anthropic(model="claude-v1-100k", temperature=0)
        elif llm_type == llm_type.VICUNA_13B:
            self.__model = Replicate(model="replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e",
                input={"temperature": 0.75, "max_length": 3000, "top_p": 0.25})
        elif llm_type == llm_type.MOSAIC:
            self.__model = MosaicML(inject_instruction_format=True,model_kwargs={'do_sample': False, 'max_length': 3000})
        else:
            raise RuntimeError("Unable to instantiate LLM model")

    def get_model(self) -> LLM:
        return self.__model
