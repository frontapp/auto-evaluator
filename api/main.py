import json
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from modules.test import run

from modules.evaluator import run_evaluator

# ===============
# API setup
# ===============
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "https://evaluator-ui.vercel.app/"
    "https://evaluator-ui.vercel.app"
    "evaluator-ui.vercel.app/"
    "evaluator-ui.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============
# API routes
# ===============
@app.get("/")
async def root():
    return {"message": "Welcome to the Auto Evaluator!"}


@app.post("/evaluator-stream")
async def create_response(
    files: List[UploadFile] = File(...),
    num_eval_questions: int = Form(5),
    chunk_chars: int = Form(1000),
    overlap: int = Form(100),
    split_method: str = Form("RecursiveTextSplitter"),
    retriever_type: str = Form("similarity-search"),
    embeddings: str = Form("OpenAI"),
    model_version: str = Form("gpt-3.5-turbo"),
    grade_prompt: str = Form("Fast"),
    num_neighbors: int = Form(3),
    test_dataset: str = Form("[]"),
):
    # Client loads in a string for the test dataset, so this is why we need to parse it.
    test_dataset = json.loads(test_dataset)
    return EventSourceResponse(run(files,
                                   num_eval_questions,
                                   chunk_chars,
                                   overlap,
                                   split_method,
                                   retriever_type,
                                   embeddings,
                                   model_version,
                                   grade_prompt,
                                   num_neighbors,
                                   test_dataset,
                                   chain_type="refine"),
                               headers={"Content-Type": "text/event-stream",
                                        "Connection": "keep-alive",
                                        "Cache-Control": "no-cache"})
