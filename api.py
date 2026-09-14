from fastapi import FastAPI

from pydantic_dataclasses import QueryRequest
from make_rag_pipeline import make_pipeline

# run with uvicorn api:app --reload

PIPELINE = make_pipeline()

app = FastAPI(title="RAG API")

@app.post("/query")
def make_query(request: QueryRequest):
    return PIPELINE.answer_query(request)
