import json
import sqlite3
from pathlib import Path
from typing import Any
import threading
import re

from extractors.base import ExtractedChunk
from .base import MetadataStoreInterface
from search_result_dataclass import SearchResult


class SQLiteMetadataStore(MetadataStoreInterface):
    def __init__(self, db_path: str | Path = "rag_vectors.db", score_threshold = -10):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # to allow each thread to access independently
        # we let each thread have its own 'self._local' 
        # so it can hold a conn object within it.
        self._local = threading.local()
        self._create_schema()
        self._setup_fts()
        self.score_threshold = score_threshold

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
            self._insert_fts_record(chunk_id, chunk.content)
            chunk_ids.append(chunk_id)

        self.conn.commit()
        return chunk_ids

    def search_by_chunk_ids(self, chunk_ids: list[int]) -> list[SearchResult]:
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

        matches = []
        for row in rows:
            matches.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    source_type=row["source_type"],
                    context=row["content"],
                    locator=json.loads(row["locator_json"] or "{}"),
                    file_name=row["file_name"],
                    metadata=json.loads(row["metadata_json"] or "{}")
                )
            )
        return matches

    def _setup_fts(self):
        """Run once during initialization to create and populate the FTS table."""
        self.conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS context_fts USING fts5(id, context_body);
            INSERT INTO context_fts (id, context_body)
            SELECT CAST(c.chunk_id AS TEXT), c.content
            FROM chunk_records AS c
            WHERE NOT EXISTS (
                SELECT 1 FROM context_fts AS f WHERE f.id = CAST(c.chunk_id AS TEXT)
            );
        """)
        self.conn.commit()

    def _insert_fts_record(self, chunk_id: int, content: str) -> None:
        self.conn.execute(
            "INSERT INTO context_fts (id, context_body) VALUES (?, ?)",
            (str(chunk_id), content),
        )

    def sparse_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Aim to find exact keywords in context, good for acronyms and esoteric words that 
        would otherwise be missed in a dense vector search. 
        
        Note: that sql bm25 has most negative as the best match"""
        safe_query: str = ' OR '.join(re.sub(r'[^\w\s]', '', query).split())
        if not safe_query:
            return []

        sql = """
            SELECT 
                id, 
                context_body, 
                bm25(context_fts) AS score
            FROM context_fts
            WHERE context_fts MATCH ?
            ORDER BY score
            LIMIT ?;
        """
        
        rows = self.conn.execute(sql, (safe_query, top_k)).fetchall()
        
        return [
            SearchResult(
                chunk_id = int(m['id']),
                sparse_rank = int(i),
                key_word_score = float(-m['score']), # the result is -ve
                context = m['context_body']
            )
            for i, m in enumerate(rows)
        ]
