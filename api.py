from fastapi import FastAPI
from pydantic import BaseModel

from make_rag_pipeline import make_pipeline

# run with uvicorn api:app --reload

# hf files in ~/.cache/huggingface/hub

PIPELINE = make_pipeline()

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def make_query(request: QueryRequest):
    return PIPELINE.answer_query(request.query)
