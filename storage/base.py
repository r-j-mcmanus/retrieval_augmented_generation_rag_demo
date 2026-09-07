from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from extractors.base import ExtractedChunk
from search_result_dataclass import SearchResult


class VectorStoreInterface(ABC):
    """Vector index backend for similarity search."""

    @abstractmethod
    def add_vectors(self, chunk_ids: list[int], vectors: Any) -> None:
        """Store embeddings for the given chunk ids."""

    @abstractmethod
    def search(self, query_vector: Any, top_k: int = 3) -> list[SearchResult]:
        """Return matches from the vector index."""


class MetadataStoreInterface(ABC):
    """Relational metadata backend. This is intentionally separate from vector search."""

    @abstractmethod
    def insert_document(
        self,
        file_path: str | Path,
        source_type: str,
        metadata: dict[str, Any],
        chunks: list[ExtractedChunk],
    )  -> list[int]:
        """Persist metadata about files and chunks."""

    @abstractmethod
    def search_by_chunk_ids(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        """Return rows for the matched chunk ids, including file metadata."""

    @abstractmethod
    def sparse_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Performs a sparse context search on the metadata"""