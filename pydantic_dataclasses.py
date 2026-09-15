from enum import StrEnum
from pydantic import BaseModel, ConfigDict, model_validator
from datetime import datetime
from typing import Any

class DocumentScope(StrEnum):
    CLIENT = "client"
    INTERNAL = "internal"
    ALL_CLIENTS = "all_clients"


class QueryRequest(BaseModel):
    """The request json the api expects"""
    query: str
    scope: DocumentScope
    client_reference: int | None = None
    document_ids: set[int] | None = None
    created_before: datetime | None = None
    created_after: datetime | None = None

    def __str__(self) -> str:
        return self.model_dump_json()

    @model_validator(mode="after")
    def validate_scope(self):
        if self.scope == DocumentScope.CLIENT:
            if self.client_reference is None:
                raise ValueError(
                    "client_reference is required for client scope"
                )
        elif self.client_reference is not None:
            raise ValueError(
                "client_reference is only valid for client scope"
            )

        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after > self.created_before
        ):
            raise ValueError(
                "created_after must be before created_before"
            )

        return self


class DocumentFilter(BaseModel):
    """Holds the state for which documents can be searched"""
    # ensure it cannot be modified after creation
    model_config = ConfigDict(frozen=True)

    client_reference: int | None = None
    include_internal: bool = True
    # example for later dev
    client_references: frozenset[int] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    response: str
    token_usage: TokenUsage


from dataclasses import dataclass, field

@dataclass
class SearchResult:
    chunk_id: int
    
    dense_rank: int = -1
    sparse_rank: int = -1
    vector_distance: float = -1.0
    key_word_score: float = 1.0
    doc_id: int = -1
    source_type: str = ''
    context: str = ''
    file_name: str = ''
    score: float = 0

    locator: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __or__(self, other: "SearchResult") -> "SearchResult":
        if not isinstance(other, SearchResult):
            return NotImplemented
        
        self.source_type = other.source_type if not self.source_type else self.source_type
        self.context = other.context if not self.context else self.context
        self.locator = other.locator if not self.locator else self.locator
        self.file_name = other.file_name if not self.file_name else self.file_name
        self.metadata = other.metadata if not self.metadata else self.metadata
        self.doc_id = other.doc_id if not self.doc_id else self.doc_id

        return self



@dataclass
class ExtractedChunk:
    """Contains a row of data for a text chunk"""
    content: str
    locator: dict[str, Any] = field(default_factory=dict)
    entity_id: int | None = None
    source_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
