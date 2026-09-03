from rag_pipeline import RAGPipeline
from extractors import BaseDocumentExtractor, PDFExtractor, VTTExtractor, MP3Extractor, HTMLExtractor
from storage import SQLiteMetadataStore, UsearchVectorStore
from embedding import BGEEmbeddingService
from llm_caller import LocalQwenLLMCaller
from fastapi import FastAPI
from pydantic import BaseModel

# run with uvicorn api:app --reload

# hf files in ~/.cache/huggingface/hub


def _index_data(pipeline: RAGPipeline):
    # probably best to make a queue trigger that can process files in blob storage as prompted by the queue
    pipeline.index_file(r'_data/vtt/example_video_1.vtt')
    pipeline.index_file(r'_data/vtt/example_video_2.vtt')
    pipeline.index_file(r'_data/pdf/example_pdf_1.pdf')
    pipeline.index_file(r'_data/mp3/example_mp3_1.mp3')


def _make_pipeline() -> RAGPipeline:
    pdf_extractor = PDFExtractor()
    vtt_extractor = VTTExtractor()
    mp3_extractor = MP3Extractor()
    html_extractor = HTMLExtractor()

    extractors: list[BaseDocumentExtractor] = [pdf_extractor, vtt_extractor, mp3_extractor, html_extractor]

    embedding_service = BGEEmbeddingService()

    sql_store = SQLiteMetadataStore("_database/rag_vectors.db")
    vector_store = UsearchVectorStore("_database/vector_index.usearch", embedding_dim=embedding_service.embedding_dim)

    llm_caller = LocalQwenLLMCaller()

    pipeline = RAGPipeline(
        extractors=extractors,
        metadata_store=sql_store,
        vector_store=vector_store,
        encoder=embedding_service,
        llm_caller=llm_caller
    )

    return pipeline

PIPELINE = _make_pipeline()
# _index_data(PIPELINE)

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def make_query(request: QueryRequest):
    return PIPELINE.answer_query(request.query)
