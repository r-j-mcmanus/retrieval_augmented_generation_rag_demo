from pathlib import Path
import os

from extractors import BaseDocumentExtractor
from storage import MetadataStoreInterface, VectorStoreInterface
from embedding import EmbeddingServiceInterface
from text_preprocessing import TextPreprocessor
from knowledge_graph import KnowledgeGraph

from pydantic_dataclasses import IndexRequest

# https://www.reddit.com/r/Rag/comments/1rf7xf6/whats_your_experience_with_hybrid_retrieval/

# TODO would be good to fine tune a model for this!

LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8001")


class RAGIngestion:
    def __init__(
        self,
        extractors: list[BaseDocumentExtractor],
        metadata_store: MetadataStoreInterface,
        vector_store: VectorStoreInterface,
        encoder: EmbeddingServiceInterface,
        preprocessor: TextPreprocessor,
        knowledge_graph: KnowledgeGraph,
        call_llm
    ):
        # ensure only valid extractors
        self.extractors: dict[str, BaseDocumentExtractor] = {}
        for extractor in extractors:
            if extractor.source_type.lower() in self.extractors:
                raise ValueError(f'Same source_type in two extractors: {extractor.source_type}')
            self.extractors[extractor.source_type.lower()] =  extractor

        self.encoder = encoder
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.preprocessor = preprocessor
        self.knowledge_graph = knowledge_graph
        self.call_llm = call_llm

    def _get_extractor_for_file(self, file_path: str | Path) -> BaseDocumentExtractor:
        suffix = Path(file_path).suffix.lower().lstrip(".")
        extractor = self.extractors.get(suffix)
        if extractor is None:
            raise ValueError(f"No extractor registered for file type: {suffix}")
        return extractor

    def index_file(self, request: IndexRequest):
        file_path = Path(request.file_path)
        extractor = self._get_extractor_for_file(file_path)
        chunks, useful_metadata = extractor.extract_document(
            file_path,
            additional_metadata={"encoder": self.encoder.name},
        )

        self.knowledge_graph.add_document(
            chunks,
            self.call_llm,
            client_reference=request.client_reference,
        )

        return []

        chunk_ids = self.metadata_store.insert_document(
            request=request,
            source_type=extractor.source_type,
            metadata=useful_metadata, 
            chunks=chunks
        )

        chunk_contents = (self.preprocessor(chunk.content) for chunk in chunks)
        encoded_chunk = self.encoder.encode_documents(chunk_contents)

        self.vector_store.add_vectors(chunk_ids=chunk_ids, vectors=encoded_chunk)
        return chunk_ids
