from abc import ABC
from pathlib import Path
from typing import Any, Iterable

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
        client_reference: int | str | None,
        metadata: dict[str, Any],
        chunks: list[ExtractedChunk],
        tags: Iterable[str] | None = None,
        visibility: str | None = None,
    )  -> list[int]:
        """Placeholder implementation for Snowflake metadata persistence."""
        raise NotImplementedError("Snowflake metadata integration not implemented yet.")

    def search_by_chunk_ids(
        self,
        chunk_ids: list[int],
        client_reference: int | str | None = None,
    ) -> list[dict[str, Any]]:
        """Placeholder implementation for Snowflake metadata lookup."""
        raise NotImplementedError("Snowflake metadata lookup not implemented yet.")

    def get_chunk_ids_for_client_reference(self, client_reference: int) -> set[int]:
        raise NotImplementedError("Snowflake client reference lookup not implemented yet.")

    def list_tags(self) -> list[str]:
        raise NotImplementedError("Snowflake tag lookup not implemented yet.")

    def add_vectors(self, chunk_ids: list[int], vectors: Any) -> None:
        """Placeholder implementation for Snowflake vector insertion."""
        raise NotImplementedError("Snowflake vector insertion not implemented yet.")

    def search(
        self,
        query_vector: Any,
        top_k: int = 3,
        allowed_chunk_ids: set[int] | None = None,
    ):
        """Placeholder implementation for Snowflake vector similarity search."""
        raise NotImplementedError("Snowflake vector search not implemented yet.")
