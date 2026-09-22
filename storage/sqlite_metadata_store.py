import json
import sqlite3
from pathlib import Path
from typing import Any, LiteralString
import threading
import re

from .base import MetadataStoreInterface
from pydantic_dataclasses import QueryRequest, DocumentFilter, SearchResult, ExtractedChunk, IndexRequest


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
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                document_type TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'internal',
                created_at TEXT,
                created_by TEXT,
                metadata_json TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_clients (
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                client_ref TEXT NOT NULL,
                PRIMARY KEY (document_id, client_ref)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_tags (
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
                PRIMARY KEY (document_id, tag_id)
            )
            """
        )

        # Keep databases created before the normalized schema usable.
        if self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'file_metadata'"
        ).fetchone():
            self.conn.execute(
                """
                INSERT OR IGNORE INTO documents
                    (id, title, document_type, visibility, created_at, created_by, metadata_json)
                SELECT doc_id, file_name, 'unknown',
                    CASE WHEN client_reference IS NULL THEN 'internal' ELSE 'client' END,
                    file_created_at, created_by, metadata_json
                FROM file_metadata
                """
            )
            self.conn.execute(
                """
                INSERT OR IGNORE INTO document_clients (document_id, client_ref)
                SELECT doc_id, CAST(client_reference AS TEXT)
                FROM file_metadata
                WHERE client_reference IS NOT NULL
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
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )
        self.conn.commit()

    def insert_document(
        self,
        request: IndexRequest,
        source_type: str,
        metadata: dict[str, Any],
        chunks: list[ExtractedChunk],
    ) -> list[int]:
        """Add one document, its relationships, and chunks."""
        file_path = Path(request.file_path)
        doc_id = self.conn.execute(
            """
            INSERT INTO documents
                (title, content, document_type, visibility, created_at, created_by, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_path.name,
                metadata.get("content", ""),
                source_type,
                request.visibility or ("client" if request.client_reference is not None else "internal"),
                metadata.get("created_at"),
                metadata.get("created_by"),
                json.dumps(metadata, default=str),
            ),
        ).lastrowid
        assert isinstance(doc_id, int)

        if request.client_reference is not None:
            self.conn.execute(
                "INSERT INTO document_clients (document_id, client_ref) VALUES (?, ?)",
                (doc_id, str(request.client_reference)),
            )

        for tag_name in request.tags or ():
            normalized_tag = tag_name.strip()
            if not normalized_tag:
                continue
            self.conn.execute(
                "INSERT OR IGNORE INTO tags (name) VALUES (?)",
                (normalized_tag,),
            )
            tag_id = self.conn.execute(
                "SELECT id FROM tags WHERE name = ?",
                (normalized_tag,),
            ).fetchone()[0]
            self.conn.execute(
                "INSERT OR IGNORE INTO document_tags (document_id, tag_id) VALUES (?, ?)",
                (doc_id, tag_id),
            )

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

    def list_tags(self) -> list[str]:
        rows = self.conn.execute("SELECT name FROM tags ORDER BY name COLLATE NOCASE").fetchall()
        return [str(row["name"]) for row in rows]

    def search_by_chunk_ids(
        self,
        chunk_ids: list[int],
        client_reference: int | str | None = None,
    ) -> list[SearchResult]:
        if not chunk_ids:
            return []

        placeholders = ", ".join("?" for _ in chunk_ids) # n ? for chunk_ids to parse into
        parameters: list[Any] = list(chunk_ids)
        client_filter = ""
        if client_reference is not None:
            client_filter = " AND EXISTS (SELECT 1 FROM document_clients dc WHERE dc.document_id = d.id AND dc.client_ref = ?)"
            parameters.append(str(client_reference))

        rows = self.conn.execute(
            f"""
            SELECT 
                c.chunk_id,
                c.doc_id,
                c.source_type,
                c.content,
                c.locator_json,
                d.title,
                d.metadata_json,
                (SELECT dc.client_ref FROM document_clients dc WHERE dc.document_id = d.id LIMIT 1) AS client_reference,
                (SELECT group_concat(t.name, ',') FROM document_tags dt JOIN tags t ON t.id = dt.tag_id WHERE dt.document_id = d.id) AS tag_names
            FROM chunk_records as c
            JOIN documents d ON d.id = c.doc_id
            WHERE c.chunk_id IN ({placeholders})
            {client_filter}
            """,
            parameters,
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
                    file_name=row["title"],
                    metadata=json.loads(row["metadata_json"] or "{}") | {
                        "client_reference": row["client_reference"],
                        "client_references": self._client_references(row["doc_id"]),
                        "tags": row["tag_names"].split(",") if row["tag_names"] else [],
                    }
                )
            )
        return matches

    @staticmethod
    def _document_filter_sql(
        document_filter: DocumentFilter,
    ) -> tuple[list[str], list[Any]]:
        """Make sql conditions based on values in the document filter"""
        conditions = ["1 = 1"]
        parameters: list[Any] = []

        if document_filter.client_reference is not None:
            conditions.append("EXISTS (SELECT 1 FROM document_clients dc WHERE dc.document_id = d.id AND dc.client_ref = ?)")
            parameters.append(str(document_filter.client_reference))
        elif not document_filter.include_internal:
            conditions.append("EXISTS (SELECT 1 FROM document_clients dc WHERE dc.document_id = d.id)")

        if document_filter.client_references is not None:
            if not document_filter.client_references:
                conditions.append("1 = 0")
            else:
                placeholders = ", ".join("?" for _ in document_filter.client_references)
                conditions.append(f"EXISTS (SELECT 1 FROM document_clients dc WHERE dc.document_id = d.id AND dc.client_ref IN ({placeholders}))")
                parameters.extend(str(ref) for ref in document_filter.client_references)

        if document_filter.created_after is not None:
            conditions.append("d.created_at >= ?")
            parameters.append(document_filter.created_after.isoformat())

        if document_filter.created_before is not None:
            conditions.append("d.created_at <= ?")
            parameters.append(document_filter.created_before.isoformat())

        return conditions, parameters

    def get_chunk_ids(self, document_filter: DocumentFilter) -> set[int]:
        conditions, parameters = self._document_filter_sql(document_filter)
        rows = self.conn.execute(
            f"""
            SELECT c.chunk_id
            FROM chunk_records AS c
            JOIN documents AS d ON d.id = c.doc_id
            WHERE {" AND ".join(conditions)}
            """,
            parameters,
        ).fetchall()
        return {int(row["chunk_id"]) for row in rows}

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

    @staticmethod
    def _build_fts_query(query: str) -> str:
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        return " OR ".join(f'"{term}"' for term in terms)

    def sparse_search(
        self,
        query_request: QueryRequest,
        allowed_chunk_ids: set[int],
        top_k: int
    ) -> list[SearchResult]:
        """Aim to find exact keywords in context, good for acronyms and esoteric words that 
        would otherwise be missed in a dense vector search. 
        
        Note: that sql bm25 has most negative as the best match"""
        query = query_request.query
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        chunk_ids = sorted(allowed_chunk_ids)
        placeholders = ", ".join("?" for _ in chunk_ids)
        
        sql = f"""
            SELECT 
                context_fts.id,
                context_fts.context_body,
                bm25(context_fts) AS score
            FROM context_fts
            JOIN chunk_records AS c ON c.chunk_id = CAST(context_fts.id AS INTEGER)
            JOIN documents AS d ON d.id = c.doc_id
            WHERE context_fts MATCH ?
            AND c.chunk_id IN ({placeholders}) 
            ORDER BY score
            LIMIT ?;
        """
        parameters = [fts_query, *chunk_ids, top_k]
        rows = self.conn.execute(sql, parameters).fetchall()

        results = [
            SearchResult(
                chunk_id = int(m['id']),
                sparse_rank = int(i),
                key_word_score = float(-m['score']), # the result is -ve
                context = m['context_body']
            )
            for i, m in enumerate(rows)
        ]

        return results

    def _sparse_search_internal(
        self,
        query_request: QueryRequest,
        top_k: int
    ) -> list[SearchResult]:
        """Aim to find exact keywords in context, good for acronyms and esoteric words that 
        would otherwise be missed in a dense vector search. 
        
        Note: that sql bm25 has most negative as the best match"""
        fts_query = self._build_fts_query(query_request.query)
        if not fts_query:
            return []

        sql: LiteralString = f"""
            SELECT 
                context_fts.id,
                context_fts.context_body,
                bm25(context_fts) AS score
            FROM context_fts
            JOIN chunk_records AS c ON c.chunk_id = CAST(context_fts.id AS INTEGER)
            JOIN documents AS d ON d.id = c.doc_id
            WHERE context_fts MATCH ?
            AND NOT EXISTS (SELECT 1 FROM document_clients dc WHERE dc.document_id = d.id)
            ORDER BY score
            LIMIT ?;
        """
        rows = self.conn.execute(sql, [fts_query, top_k]).fetchall()

        return [
            SearchResult(
                chunk_id = int(m['id']),
                sparse_rank = int(i),
                key_word_score = float(-m['score']), # the result is -ve
                context = m['context_body']
            )
            for i, m in enumerate(rows)
        ]

    def _client_references(self, document_id: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT client_ref FROM document_clients WHERE document_id = ? ORDER BY client_ref",
            (document_id,),
        ).fetchall()
        return [str(row["client_ref"]) for row in rows]
