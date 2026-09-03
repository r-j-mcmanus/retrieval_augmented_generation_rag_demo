import json
import sqlite3
from pathlib import Path
from typing import Any
import threading

from extractors.base import ExtractedChunk
from .base import MetadataStoreInterface


class SQLiteMetadataStore(MetadataStoreInterface):
    def __init__(self, db_path: str | Path = "rag_vectors.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # to allow each thread to access independently
        # we let each thread have its own 'self._local' 
        # so it can hold a conn object within it.
        self._local = threading.local()
        self._create_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            # Created once per thread when that thread first queries
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row  # Optional: return dict-like rows
        return self._local.conn

    def _create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_metadata (
                doc_id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_created_at TEXT,
                created_by TEXT,
                metadata_json TEXT
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_records (
                chunk_id INTEGER PRIMARY KEY,  -- Global surrogate key for every chunk
                doc_id INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                content TEXT NOT NULL,
                locator_json TEXT,
                metadata_json TEXT,
                FOREIGN KEY (doc_id) REFERENCES file_metadata(doc_id) ON DELETE CASCADE
            )
            """
        )
        self.conn.commit()

    def insert_document(
        self,
        file_path: str | Path,
        source_type: str,
        metadata: dict[str, Any],
        chunks: list[ExtractedChunk],
    ) -> list[int]:
        """Add one file to the file_metadata table and get the row id"""
        file_path = Path(file_path)

        doc_id = self.conn.execute(
            """
            INSERT INTO file_metadata (file_name, file_created_at, created_by, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                file_path.name,
                metadata.get("created_at"),
                metadata.get("created_by"),
                json.dumps(metadata, default=str),
            ),
        ).lastrowid
        assert isinstance(doc_id, int)

        chunk_ids: list[int] = []
        for chunk in chunks:
            chunk_id = self.conn.execute(
                """
                INSERT INTO chunk_records (doc_id, source_type, content, locator_json, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    source_type,
                    chunk.content,
                    json.dumps(chunk.locator, default=str),
                    json.dumps(chunk.metadata, default=str),
                ),
            ).lastrowid
            assert isinstance(chunk_id, int)
            chunk_ids.append(chunk_id)

        self.conn.commit()
        return chunk_ids

    def search_by_chunk_ids(self, chunk_ids: list[int]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []

        placeholders = ", ".join("?" for _ in chunk_ids) # n ? for chunk_ids to parse into
        rows = self.conn.execute(
            f"""
            SELECT 
                c.chunk_id,
                c.doc_id,
                c.source_type,
                c.content,
                c.locator_json,
                f.file_name,
                f.metadata_json
            FROM chunk_records as c
            JOIN file_metadata f ON f.doc_id = c.doc_id
            WHERE c.chunk_id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()

        matches: list[dict[str, Any]] = []
        for row in rows:
            matches.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "source_type": row["source_type"],
                    "content": row["content"],
                    "locator": json.loads(row["locator_json"] or "{}"),
                    "file_name": row["file_name"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            )
        return matches
