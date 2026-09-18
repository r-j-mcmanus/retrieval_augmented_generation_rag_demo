from fastapi import FastAPI

from pydantic_dataclasses import QueryRequest, QueryResponse
from make_rag_pipeline import make_pipeline

# run with uvicorn rag_api:app --reload

PIPELINE = make_pipeline()

app = FastAPI(title="RAG API")

@app.post("/query", response_model=QueryResponse)
def make_query(request: QueryRequest) -> QueryResponse:
    return PIPELINE.answer_query(request)
