from enum import Enum
from typing import List, Union

from langchain.schema import Document
from langchain.retrievers import SVMRetriever, TFIDFRetriever
from langchain.vectorstores.base import VectorStoreRetriever
from modules.text_splitter import TextSplitter

from modules.embeddings import EvaluatorEmbeddingModel
from langchain.vectorstores import FAISS

EvaluatorRetrieverType = Union[SVMRetriever, TFIDFRetriever, VectorStoreRetriever]

# The algorithm to use for splitting texts
class EvaluatorRetrieverEnum(Enum):
    SIMILARITY_SEARCH = "similarity-search"
    SVM = "SVM"
    TF_IDF = "TF-IDF"
    ANTHROPIC_100K = "Anthropic-100k"

class EvaluatorRetriever:
    __retriever_type: EvaluatorRetrieverEnum
    __retriever: EvaluatorRetrieverType

    def __init__(self, retriever_type: EvaluatorRetrieverEnum):
        if retriever_type not in EvaluatorRetrieverEnum:
            raise ValueError("Invalid retriever type")

        self.__retriever_type = retriever_type

    def index_documents(self,
                        docs: List[Document],
                        splitter: TextSplitter,
                        embedding_model: EvaluatorEmbeddingModel,
                        num_neighbors: int):

        # just extract the page content
        split_docs = splitter.split_text(" ".join(map(lambda doc: doc.page_content, docs)))

        if self.__retriever_type == EvaluatorRetrieverEnum.SIMILARITY_SEARCH:
            vectorstore = FAISS.from_documents(docs, embedding_model)
            self.__retriever = vectorstore.as_retriever(k=num_neighbors)
        elif self.__retriever_type == EvaluatorRetrieverEnum.SVM:
            self.__retriever = SVMRetriever.from_texts(split_docs, embedding_model)
        elif self.__retriever_type == EvaluatorRetrieverEnum.TF_IDF:
            self.__retriever = TFIDFRetriever.from_texts(split_docs)
        elif self.__retriever_type == EvaluatorRetrieverEnum.ANTHROPIC_100K:
            raise NotImplementedError("Anthropic-100k retriever not implemented yet")
        else:
            raise RuntimeError("Unable to instantiate retriever")


    def index_texts(self,
                    texts: List[str],
                    embedding_model: EvaluatorEmbeddingModel,
                    num_neighbors: int):
        if self.__retriever_type == EvaluatorRetrieverEnum.SIMILARITY_SEARCH:
            vectorstore = FAISS.from_texts(texts, embedding_model)
            self.__retriever = vectorstore.as_retriever(k=num_neighbors)
        elif self.__retriever_type == EvaluatorRetrieverEnum.SVM:
            self.__retriever = SVMRetriever.from_texts(texts, embedding_model)
        elif self.__retriever_type == EvaluatorRetrieverEnum.TF_IDF:
            self.__retriever = TFIDFRetriever.from_texts(texts)
        elif self.__retriever_type == EvaluatorRetrieverEnum.ANTHROPIC_100K:
            raise NotImplementedError("Anthropic-100k retriever not implemented yet")
        else:
            raise RuntimeError("Unable to instantiate retriever")

    def get_retriever(self) -> EvaluatorRetrieverType:
        return self.__retriever


