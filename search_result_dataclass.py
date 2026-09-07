
from dataclasses import dataclass, field, fields

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
        self.file_name = other.file_name if not self.source_type else self.source_type
        self.metadata = other.metadata if not self.metadata else self.metadata
        self.doc_id = other.doc_id if not self.doc_id else self.doc_id

        return self