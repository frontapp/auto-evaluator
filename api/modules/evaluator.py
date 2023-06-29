import json
import logging
import time
from typing import List

import pandas as pd
from fastapi import UploadFile

from helpers.text_helpers import parse_files
from modules.embeddings import EvaluatorEmbeddingEnum, EvaluatorEmbeddings
from modules.evaluator_llm import EvaluatorLLM, EvaluatorLLMEnum
from modules.grader_chain import GradeAnswerTypeEnum, GradeDocsTypeEnum, GraderChain
from modules.question_answer_chain import QuestionAnswerChain, QuestionAnswerChainEnum
from modules.retriever import EvaluatorRetriever, EvaluatorRetrieverEnum
from modules.text_splitter import SplitterTypeEnum, TextSplitter


def run(files: List[UploadFile],
        num_eval_questions: int,
        chunk_chars: int,
        overlap: int,
        split_method: str,
        retriever_type: str,
        embeddings: str,
        model_version: str,
        grade_prompt: str,
        num_neighbors: int,
        test_dataset: str,
        chain_type: str,
        ):
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    texts, docs = parse_files(files, logger)

    # Split all the text into manageable chunks, as passed in by the client.
    # TODO overlap is hard-coded rn
    text_splitter = TextSplitter(SplitterTypeEnum(split_method), chunk_chars, 1)
    split_text = text_splitter.split_text(" ".join(texts))
    split_docs = text_splitter.split_text(" ".join(map(lambda doc: doc.page_content, docs))) # just extract the page content

    # Create the Retriever
    logger.info("Creating the Retriever")
    evaluator_retriever = EvaluatorRetriever(retriever_type=EvaluatorRetrieverEnum(retriever_type))

    # Create the embeddings model.
    logger.info("Creating the Embeddings model")
    evaluator_embeddings = EvaluatorEmbeddings(embedding_type=EvaluatorEmbeddingEnum(embeddings))

    # Create the index
    logger.info("Creating the index")
    if len(split_docs) > 0:
        evaluator_retriever.index_documents(docs=docs, splitter=text_splitter, embedding_model=evaluator_embeddings.get_embedding_model(), num_neighbors=num_neighbors)
    else:
        evaluator_retriever.index_texts(split_text, evaluator_embeddings.get_embedding_model(), num_neighbors=num_neighbors)

    # Create the LLM
    logger.info("Creating the LLM")
    evaluator_llm = EvaluatorLLM(llm_type=EvaluatorLLMEnum(model_version))

    # Create the QA chain
    logger.info("Creating the QA chain")
    question_answer_chain = QuestionAnswerChain(llm=evaluator_llm.get_model(),
                                                retriever=evaluator_retriever.get_retriever(),
                                                chain_type=QuestionAnswerChainEnum(chain_type))

    # Create the evaluator chain
    # TODO FIX THIS 
    """ if grade_prompt in GradeDocsTypeEnum: """
    """     doc_grade_prompt = GradeDocsTypeEnum(grade_prompt) """
    """ else: """
    """     doc_grade_prompt = GradeDocsTypeEnum.DEFAULT """
    """"""
    grader_chain = GraderChain(grade_answer_type=GradeAnswerTypeEnum(grade_prompt),
                               grade_docs_type=GradeDocsTypeEnum.DEFAULT)

    generated_answers = []

    # ===========================
    # Run the QA chain!
    # ===========================
    logger.info("Running the QA chain")
    for i in range(num_eval_questions):
        if i < len(test_dataset):
            question_answer_pair = test_dataset[i]
        else:
            break

        doc_links = []

        start_time = time.time()

        # Generate the answer from the question
        generated_answer = question_answer_chain.call_chain(query=question_answer_pair["question"])
        generated_answers.append(generated_answer)

        for doc in generated_answer["source_documents"]:
            if (doc.metadata):
                logger.info("Found metadata for doc: " + doc.metadata["url"])
                doc_links.append(doc.metadata["url"])

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Grade the answer & retrieval docs
        graded_answer = grader_chain.grade_answer([test_dataset[i]], [generated_answer])
        graded_docs = grader_chain.grade_retrieval([test_dataset[i]], [generated_answer])

        graded_answer_text = graded_answer[0]["text"]
        graded_doc_text = graded_docs[0]["text"]

        # Output the response to the client
        output = {
            "question": question_answer_pair["question"],
            "answer": question_answer_pair["answer"],
            "result": generated_answer["result"],
            "docLinks": doc_links,
            "latency": elapsed_time,
            "answerScore": {"score": 1 if "Incorrect" not in graded_answer_text else 0,
                             "justification": graded_answer_text},
            "retrievalScore": {"score": 1 if "Incorrect" not in graded_doc_text else 0,
                             "justification": graded_doc_text}
        }

        api_response = generate_api_response([output])
        d_dict = api_response.to_dict('records')
        if len(d_dict) == 1:
            # This is where we return items one by one.
            yield json.dumps({"data":  d_dict[0]})
        else:
            logger.warn(
                "A QA pair was not evaluated correctly. Skipping this pair.")

def generate_api_response(output):
    d = pd.DataFrame(output)

    return d
