from abc import ABC
from pathlib import Path
from typing import Any

from extractors.base import ExtractedChunk
from .base import MetadataStoreInterface, VectorStoreInterface


class SnowflakeCombinedStore(MetadataStoreInterface, VectorStoreInterface):
    """Example Snowflake-backed metadata store for relational metadata."""

    def __init__(self, account: str, user: str, password: str, database: str, schema: str, warehouse: str):
        super().__init__()
        self.account = account
        self.user = user
        self.password = password
        self.database = database
        self.schema = schema
        self.warehouse = warehouse

    def insert_document(
        self,
        file_path: str | Path,
        source_type: str,
        metadata: dict[str, Any],
        chunks: list[ExtractedChunk],
    )  -> list[int]:
        """Placeholder implementation for Snowflake metadata persistence."""
        raise NotImplementedError("Snowflake metadata integration not implemented yet.")

    def search_by_chunk_ids(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        """Placeholder implementation for Snowflake metadata lookup."""
        raise NotImplementedError("Snowflake metadata lookup not implemented yet.")

    def add_vectors(self, chunk_ids: list[int], vectors: Any) -> None:
        """Placeholder implementation for Snowflake vector insertion."""
        raise NotImplementedError("Snowflake vector insertion not implemented yet.")

    def search(self, query_vector: Any, top_k: int = 3):
        """Placeholder implementation for Snowflake vector similarity search."""
        raise NotImplementedError("Snowflake vector search not implemented yet.")
