from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pydantic_dataclasses import QueryRequest, QueryResponse, IndexRequest
from make_rag_pipeline import make_pipeline

# run with uvicorn api_rag:app --port 8000 --reload

PIPELINE = make_pipeline()

app = FastAPI(title="RAG API")


class IndexResponse(BaseModel):
    filepath: str
    visibility: Literal["client", "internal"]
    client_reference: str | None = None
    tags: list[str]
    chunk_count: int


@app.get("/tags", response_model=list[str])
def list_tags() -> list[str]:
    return PIPELINE.metadata_store.list_tags()


@app.post("/index", response_model=IndexResponse)
def index_document(request: IndexRequest) -> IndexResponse:
    file_path = Path(request.file_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File does not exist: {request.file_path}")

    chunk_ids = PIPELINE.index_file(request)

    return IndexResponse(
        filepath=str(file_path),
        visibility=request.visibility,
        client_reference=request.client_reference,
        tags=request.tags,
        chunk_count=len(chunk_ids),
    )

@app.post("/query", response_model=QueryResponse)
def make_query(request: QueryRequest) -> QueryResponse:
    return PIPELINE.answer_query(request)
