
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
