import itertools
import json
import logging
import random
import time
from json import JSONDecodeError

import pandas as pd
from helpers.prompt_utils import (
    GRADE_ANSWER_PROMPT,
    GRADE_ANSWER_PROMPT_BIAS_CHECK,
    GRADE_ANSWER_PROMPT_FAST,
    GRADE_ANSWER_PROMPT_OPENAI,
    GRADE_DOCS_PROMPT,
    GRADE_DOCS_PROMPT_FAST,
    QA_CHAIN_PROMPT,
)
from helpers.text_helpers import get_text_from_csv, get_text_from_pdf, get_text_from_txt
from langchain.chains import QAGenerationChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import (
    LlamaCppEmbeddings,
    MosaicMLInstructorEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.evaluation.qa import QAEvalChain
from langchain.llms import Anthropic, MosaicML, Replicate
from langchain.retrievers import SVMRetriever, TFIDFRetriever
from langchain.schema import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS


def run_eval(chain, retriever, eval_qa_pair, grade_prompt, retriever_type, num_neighbors, text, logger):
    """
    Runs evaluation on a model's performance on a given evaluation dataset.
    @param chain: Model chain used for answering questions
    @param retriever:  Document retriever used for retrieving relevant documents
    @param eval_set: List of dictionaries containing questions and corresponding ground truth answers
    @param grade_prompt: String prompt used for grading model's performance
    @param retriever_type: String specifying the type of retriever used
    @param num_neighbors: Number of neighbors to retrieve using the retriever
    @param text: full document text
    @return: A tuple of four items:
    - answers_grade: A dictionary containing scores for the model's answers.
    - retrieval_grade: A dictionary containing scores for the model's document retrieval.
    - latencies_list: A list of latencies in seconds for each question answered.
    - predictions_list: A list of dictionaries containing the model's predicted answers and relevant documents for each question.
    """

    logger.info("`Running eval ...`")
    predictions = []
    retrieved_docs = []
    gt_dataset = []
    latency = []
    doc_links = []

    # FIXME: this is hacked to work with `refine`
    docs = retriever.get_relevant_documents(eval_qa_pair["question"])

    # Get answer and log latency
    start_time = time.time()
    if retriever_type == "Anthropic-100k":
        docs = [Document(page_content=text)]
        answer = chain.run(input_documents=docs, question=eval_qa_pair["question"])
        predictions.append(
            {"question": eval_qa_pair["question"], "answer": eval_qa_pair["answer"], "result": answer})
    else:
        # FIXME this is hacked to work with `refine`. Previously, it was just
        # `predictions.append(chain(eval_qa_pair))`
        predictions.append(chain({"input_documents": docs, "question": eval_qa_pair["question"]}))
    gt_dataset.append(eval_qa_pair)
    end_time = time.time()
    elapsed_time = end_time - start_time
    latency.append(elapsed_time)

    # Extract text from retrieved docs
    retrieved_doc_text = ""
    if retriever_type == "Anthropic-100k":
        retrieved_doc_text = "Doc %s: " % str(eval_qa_pair["answer"])
    else:
        docs = retriever.get_relevant_documents(eval_qa_pair["question"])
        for i, doc in enumerate(docs):
            if (doc.metadata):
                doc_links.append(doc.metadata["url"])
            retrieved_doc_text += "Doc %s: " % str(i+1) + \
                doc.page_content + " "

    # Log
    retrieved = {"question": eval_qa_pair["question"],
                 "answer": eval_qa_pair["answer"], "result": retrieved_doc_text}
    retrieved_docs.append(retrieved)

    # Grade
    graded_answers = grade_model_answer(
        gt_dataset, predictions, grade_prompt, logger)
    graded_retrieval = grade_model_retrieval(
        gt_dataset, retrieved_docs, grade_prompt, logger)
    return graded_answers, graded_retrieval, latency, predictions, doc_links


def generate_eval(text, chunk, logger):
    """
    Generate question answer pair from input text 
    @param text: text to generate eval set from
    @param chunk: chunk size to draw question from text
    @param logger: logger
    @return: dict with keys "question" and "answer"
    """

    logger.info("`Generating eval QA pair ...`")
    # Generate random starting index in the doc to draw question from
    num_of_chars = len(text)
    starting_index = random.randint(0, num_of_chars-chunk)
    sub_sequence = text[starting_index:starting_index+chunk]
    # Set up QAGenerationChain chain using GPT 3.5 as default
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    # Catch any QA generation errors and re-try until QA pair is generated
    awaiting_answer = True
    while awaiting_answer:
        try:
            qa_pair = chain.run(sub_sequence)
            eval_set.append(qa_pair)
            awaiting_answer = False
        except JSONDecodeError:
            logger.error("Error on question")
            starting_index = random.randint(0, num_of_chars-chunk)
            sub_sequence = text[starting_index:starting_index+chunk]
    eval_pair = list(itertools.chain.from_iterable(eval_set))
    return eval_pair


def split_texts(text, chunk_size, overlap, split_method, logger):
    """
    Split text into chunks
    @param text: text to split
    @param chunk_size: charecters per split
    @param overlap: charecter overlap between splits
    @param split_method: method used to split text
    @param logger: logger
    @return: list of str splits
    """

    logger.info("`Splitting doc ...`")
    if split_method == "RecursiveTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(separator=" ",
                                              chunk_size=chunk_size,
                                              chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    return splits


def make_llm(model):
    """
    Make LLM
    @param model: LLM to use
    @return: LLM
    """

    if model in ("gpt-3.5-turbo", "gpt-4"):
        llm = ChatOpenAI(model_name=model, temperature=0)
    elif model == "anthropic":
        llm = Anthropic(temperature=0)
    elif model == "Anthropic-100k":
        llm = Anthropic(model="claude-v1-100k", temperature=0)
    elif model == "vicuna-13b":
        llm = Replicate(model="replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e",
                input={"temperature": 0.75, "max_length": 3000, "top_p": 0.25})
    elif model == "mosaic":
        llm = MosaicML(inject_instruction_format=True,model_kwargs={'do_sample': False, 'max_length': 3000})
    return llm

def make_retriever(splits, docs, retriever_type, embeddings, num_neighbors, llm, isUsingDocs, logger):
    """
    Make document retriever
    @param splits: list of str splits
    @param retriever_type: retriever type
    @param embedding_type: embedding type
    @param num_neighbors: number of neighbors for retrieval
    @param _llm: model
    @param logger: logger
    @return: retriever
    """

    logger.info("`Making retriever ...`")
    # Set embeddings
    if embeddings == "OpenAI":
        embd = OpenAIEmbeddings()
    elif embeddings == "SentenceTransformer":
        embd = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-distilroberta-v1")
    # Note: Still WIP (can't be selected by user yet)
    elif embeddings == "LlamaCppEmbeddings":
        embd = LlamaCppEmbeddings(model="replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e")
    # Note: Test
    elif embeddings == "Mosaic":
        embd = MosaicMLInstructorEmbeddings(query_instruction="Represent the query for retrieval: ")

    # Select retriever
    if retriever_type == "similarity-search":
        vectorstore = FAISS.from_documents(docs, embd) if isUsingDocs else FAISS.from_texts(splits, embd)
        retriever = vectorstore.as_retriever(k=num_neighbors)
    elif retriever_type == "SVM":
        retriever = SVMRetriever.from_texts(splits, embd)
    elif retriever_type == "TF-IDF":
        retriever = TFIDFRetriever.from_texts(splits)
    elif retriever_type == "Anthropic-100k":
        retriever = llm
    return retriever


def make_chain(llm, retriever, retriever_type, chain_type: str, model):
    """
    Make retrieval chain
    @param llm: model
    @param retriever: retriever
    @param retriever_type: retriever type
    @param chain_typ: chain type ('stuff', 'refine', 'map_reduce', 'map_rerank'). See: https://python.langchain.com/docs/modules/chains/additional/question_answering.html
    @return: QA chain
    """

    # Select prompt
    if model == "vicuna-13b":
        # Note: Better answer quality using default prompt
        # chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT_LLAMA}
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
    else:
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}

    # Select model
    if retriever_type == "Anthropic-100k":
        qa_chain = load_qa_chain(llm, chain_type=chain_type, prompt=QA_CHAIN_PROMPT)
    elif chain_type == 'refine':
        qa_chain = load_qa_chain(llm, chain_type=chain_type)
    else:
        qa_chain = RetrievalQA.from_chain_type(llm,
                                               chain_type=chain_type,
                                               retriever=retriever,
                                               chain_type_kwargs=chain_type_kwargs,
                                               input_key="question")
    return qa_chain


def grade_model_answer(predicted_dataset, predictions, grade_answer_prompt, logger):
    """
    Grades the answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @param logger: logger
    @return: A list of scores for the distilled answers.
    """

    logger.info("`Grading model answer ...`")
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(predicted_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


def grade_model_retrieval(gt_dataset, predictions, grade_docs_prompt, logger):
    """
    Grades the relevance of retrieved documents based on ground truth and model predictions.
    @param gt_dataset: list of dictionaries containing ground truth questions and answers.
    @param predictions: list of dictionaries containing model predictions for the questions
    @param grade_docs_prompt: prompt level for the grading.
    @return: list of scores for the retrieved documents.
    """

    logger.info("`Grading relevance of retrieved docs ...`")
    if grade_docs_prompt == "Fast":
        prompt = GRADE_DOCS_PROMPT_FAST
    else:
        prompt = GRADE_DOCS_PROMPT

    # Note: GPT-4 grader is advised by OAI
    eval_chain = QAEvalChain.from_llm(llm=ChatOpenAI(model_name="gpt-4", temperature=0),
                                      prompt=prompt)
    graded_outputs = eval_chain.evaluate(gt_dataset,
                                         predictions,
                                         question_key="question",
                                         prediction_key="result")
    return graded_outputs


# TODO Types
def run_evaluator(
    files,
    num_eval_questions,
    chunk_chars,
    overlap,
    split_method,
    retriever_type,
    embeddings,
    model_version,
    grade_prompt,
    num_neighbors,
    test_dataset
):
    """
    Runs the evaluator to be processed and shown in the client.
    @param files: Files where the content is; the "knowledge-base".
    @param num_eval_questions: Number of questions to be evaluated. If test_dataset is given, take it from there. Otherwise, generate one with an LLM.
    @param chunk_chars: Parameter for splitting the documents into chunks.
    @param overlap: Chunk overlap configuration.
    @param split_method: Determines which algorithm to use when splitting text.
    @param retriever_type: What algorithm to use for retrieving documents from a vector store.
    @param embeddings: Embedding algorithm — e.g. OpenAI's ADA, SentenceTransformer, etc.
    @param model_version: LLM Model version
    @param grade_prompt: Parameter to determine _how_ to grade the prompt; e.g. "Fast", "Full", etc.
    @param num_neighbors: Number of documents to retrieve for a given question.
    @param test_dataset: Q/A (actual question, actual answer) formatted dataset to be used for evaluation.
    """

    # Set up logging
    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    # Read content of files to be indexed.
    texts = []
    fnames = []
    docs = []
    isUsingDocs = False

    # Parse the files and extract the text into a `texts[]` or `docs[]`
    for file in files:
        logger.info("Reading file: {}".format(file.filename))
        contents = file.file.read()
        # PDF file
        if file.content_type == 'application/pdf':
            logger.info("File {} is a PDF".format(file.filename))
            pdfText = get_text_from_pdf(file, contents, logger)

            texts.append(pdfText)
            fnames.append(file.filename)
        # Text file
        elif file.content_type == 'text/plain':
            logger.info("File {} is a TXT".format(file.filename))
            txtText = get_text_from_txt(file, contents, logger)

            texts.append(txtText)
            fnames.append(file.filename)
        # CSV files
        # TODO make this more robust with document thing — currently it's just text on the whole thing
        elif file.content_type == 'text/csv':
            logger.info("File {} is a CSV".format(file.filename))
            isUsingDocs = True
            csvDocs = get_text_from_csv(contents, logger)

            docs.extend(csvDocs)
            fnames.append(file.filename)
        else:
            logger.warning(
                "Unsupported file type for file: {}".format(file.filename))

    text = " ".join(texts)
    splits = ""

    if retriever_type == "Anthropic-100k":
        model_version = "Anthropic-100k"
    elif retriever_type == "similarity-search":
        logger.info("Splitting with documents")
    else:
        logger.info("Splitting texts")
        splits = split_texts(text, chunk_chars, overlap, split_method, logger)

    logger.info("Make LLM")
    llm = make_llm(model_version)

    logger.info("Make retriever")
    retriever = make_retriever(
        splits, docs, retriever_type, embeddings, num_neighbors, llm, isUsingDocs, logger)

    logger.info("Make chain")
    qa_chain = make_chain(llm, retriever, retriever_type, 'refine', model_version)

    for i in range(num_eval_questions):
        # Run eval
        graded_answers, graded_retrieval, latency, predictions, doc_links = compute_question_answer(
                i, text, test_dataset, qa_chain, retriever, grade_prompt, retriever_type, num_neighbors, logger
                )
        # Assemble output
        d = pd.DataFrame(predictions)
        d['answerScore'] = [g['text'] for g in graded_answers]
        d['retrievalScore'] = [g['text'] for g in graded_retrieval]
        d['latency'] = latency
        d['docLinks'] = ' '.join(doc_links)

        # Summary statistics
        d['answerScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                             'justification': text} for text in d['answerScore']]
        d['retrievalScore'] = [{'score': 1 if "Incorrect" not in text else 0,
                                'justification': text} for text in d['retrievalScore']]

        # Convert dataframe to dict
        d_dict = d.to_dict('records')
        if len(d_dict) == 1:
            # This is where we return items one by one.
            yield json.dumps({"data":  d_dict[0]})
        else:
            logger.warn(
                "A QA pair was not evaluated correctly. Skipping this pair.")


def compute_question_answer(i: int, text, test_dataset, qa_chain, retriever, grade_prompt, retriever_type, num_neighbors, logger):
    # Generate Question / Answer based on given dataset
    if i < len(test_dataset):
        eval_pair = test_dataset[i]
    else:
        # Generate random Question / Answer
        eval_pair = generate_eval(text, 3000, logger)
        if len(eval_pair) == 0:
            # Error in eval generation
            return
        else:
            # This returns a list, so we unpack to dict
            eval_pair = eval_pair[0]

    graded_answers, graded_retrieval, latency, predictions, doc_links = run_eval(
        qa_chain, retriever, eval_pair, grade_prompt, retriever_type, num_neighbors, text, logger)

    return graded_answers, graded_retrieval, latency, predictions, doc_links
