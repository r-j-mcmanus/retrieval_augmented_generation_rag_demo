from typing import Any
from pathlib import Path

from usearch.index import Index
import numpy as np

from .base import VectorStoreInterface
from search_result_dataclass import SearchResult


class UsearchVectorStore(VectorStoreInterface):
    """Vector backend using Usearch for cosine similarity search."""

    def __init__(self, index_path: str | Path, embedding_dim: int, distance_threshold: float = 0.1):
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__()
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.distance_threshold = distance_threshold

        # Load existing index if it exists on disk, otherwise initialize new
        if index_path.exists():
            # view is weather it is read only or loaded into ram. Not suitable for large amounts of data!
            self.index:Index = Index.restore(self.index_path, view=False) # type: ignore
        else:
            self.index = Index(ndim=self.embedding_dim, metric="cos")

    def add_vectors(self, chunk_ids: list[int], vectors: Any) -> None:
        if not chunk_ids:
            return
        self.index.add(keys=chunk_ids, vectors=vectors)
        self.index.save(self.index_path)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        matches = self.index.search(query_vector, top_k)
        return [
            SearchResult(
                chunk_id = int(m.key),
                dense_rank = int(i),
                vector_distance = float(m.distance)
            )
            for i, m in enumerate(matches) # type: ignore
        ]
