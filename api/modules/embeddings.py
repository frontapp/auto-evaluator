from enum import Enum
from typing import Union

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import (
    LlamaCppEmbeddings,
    MosaicMLInstructorEmbeddings,
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings
)

EvaluatorEmbeddingModel = Union[OpenAIEmbeddings, HuggingFaceEmbeddings, LlamaCppEmbeddings, MosaicMLInstructorEmbeddings]

# The algorithm to use for splitting texts
class EvaluatorEmbeddingEnum(Enum):
    OPEN_AI = "OpenAI"
    SENTENCE_TRANSFORMER = "SentenceTransformer"
    LLAMA = "LlamaCppEmbeddings"
    MOSAIC = "Mosaic"

class EvaluatorEmbeddings:
    def __init__(self, embedding_type: EvaluatorEmbeddingEnum):
        if embedding_type not in EvaluatorEmbeddingEnum:
            raise ValueError("Invalid embedding type")

        if embedding_type == EvaluatorEmbeddingEnum.OPEN_AI:
            self.embedding_model = OpenAIEmbeddings()
        elif embedding_type == EvaluatorEmbeddingEnum.SENTENCE_TRANSFORMER:
            self.embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-distilroberta-v1")
        elif embedding_type == EvaluatorEmbeddingEnum.LLAMA:
            self.embedding_model = LlamaCppEmbeddings(model="replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e")
        elif embedding_type == EvaluatorEmbeddingEnum.MOSAIC:
            self.embedding_model = MosaicMLInstructorEmbeddings(query_instruction="Represent the query for retrieval: ")
        else:
            raise RuntimeError("Unable to instantiate embedding model")

    def get_embedding_model(self) -> EvaluatorEmbeddingModel:
        return self.embedding_model


