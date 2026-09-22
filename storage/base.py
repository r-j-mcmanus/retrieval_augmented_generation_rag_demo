from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


from pydantic_dataclasses import QueryRequest, DocumentFilter, SearchResult, ExtractedChunk, IndexRequest

class VectorStoreInterface(ABC):
    """Vector index backend for similarity search."""

    @abstractmethod
    def add_vectors(self, chunk_ids: list[int], vectors: Any) -> None:
        """Store embeddings for the given chunk ids."""

    @abstractmethod
    def search(
        self,
        query_vector: Any,
        top_k: int = 3,
        allowed_chunk_ids: set[int] | None = None,
    ) -> list[SearchResult]:
        """Return matches from the vector index."""


class MetadataStoreInterface(ABC):
    """Relational metadata backend. This is intentionally separate from vector search."""

    @abstractmethod
    def insert_document(
        self,
        request: IndexRequest,
        source_type: str,
        metadata: dict[str, Any],
        chunks: list[ExtractedChunk],
    )  -> list[int]:
        """Persist metadata about files and chunks."""

    @abstractmethod
    def search_by_chunk_ids(
        self,
        chunk_ids: list[int],
        client_reference: int | str | None = None,
    ) -> list[SearchResult]:
        """Return rows for the matched chunk ids, including file metadata."""

    @abstractmethod
    def get_chunk_ids(self, document_filter: DocumentFilter) -> set[int]:
        """Return all chunk IDs belonging to a client reference."""

    @abstractmethod
    def list_tags(self) -> list[str]:
        """Return existing tag names for ingestion suggestions."""

    @abstractmethod
    def sparse_search(
        self,
        query_request: QueryRequest,
        allowed_chunk_ids: set[int],
        top_k: int,
    ) -> list[SearchResult]:
        """Performs a sparse context search on the metadata"""
