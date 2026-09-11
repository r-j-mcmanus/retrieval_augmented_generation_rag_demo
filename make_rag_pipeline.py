from rag_pipeline import RAGPipeline
from extractors import BaseDocumentExtractor, PDFExtractor, VTTExtractor, MP3Extractor, HTMLExtractor, TXTExtractor, EMLExtractor
from storage import SQLiteMetadataStore, UsearchVectorStore
from embedding import BGEEmbeddingService
from llm_caller import LocalQwenLLMCaller, HFModels
from prompt_router import IntentRouter, ComparisonPath, NumericalPath, ListPath
from reranking import ReRanker
from text_preprocessing import TextPreprocessor

# hf files in ~/.cache/huggingface/hub

def _make_query_router():
    router = IntentRouter()
    router.set_paths([
        ComparisonPath(),
        NumericalPath(),
        ListPath()
    ])
    return router

def _make_extractors() -> list[BaseDocumentExtractor]:
    pdf_extractor = PDFExtractor()
    vtt_extractor = VTTExtractor()
    mp3_extractor = MP3Extractor()
    html_extractor = HTMLExtractor()
    txt_extractor = TXTExtractor()
    eml_extractor = EMLExtractor()

    return [pdf_extractor, vtt_extractor, mp3_extractor, html_extractor, txt_extractor, eml_extractor]


def make_pipeline() -> RAGPipeline:
    extractors = _make_extractors()
    router = _make_query_router()
    embedding_service = BGEEmbeddingService()

    sql_store = SQLiteMetadataStore("_database/rag_vectors.db")
    vector_store = UsearchVectorStore("_database/vector_index.usearch", embedding_dim=embedding_service.embedding_dim)

    llm_caller = LocalQwenLLMCaller(model_name=HFModels.Qwen_2_5__1_5B, temperature=0)

    re_ranker = ReRanker()
    preprocessor = TextPreprocessor()

    pipeline = RAGPipeline(
        extractors=extractors,
        metadata_store=sql_store,
        vector_store=vector_store,
        encoder=embedding_service,
        llm_caller=llm_caller,
        query_router=router,
        re_ranker=re_ranker,
        preprocessor=preprocessor
    )

    return pipeline
