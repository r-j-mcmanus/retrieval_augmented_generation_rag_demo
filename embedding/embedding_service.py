from typing import Iterable

from sentence_transformers import SentenceTransformer
import numpy as np

from .base import EmbeddingServiceInterface


class BGEEmbeddingService(EmbeddingServiceInterface):
    """Uses a BGE model and sentence transformer for the encoding."""

    def __init__(self):
        super().__init__("BAAI/bge-small-en-v1.5", 384)
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder="./_local_models") # if we wanted it local
        self.query_prefix = "Represent this sentence for searching relevant passages: "
        self.document_prefix = "Represent this document for retrieval: "

    def encode_query(self, query: str) -> np.ndarray:
        query = self.query_prefix + query
        return self.model.encode(query, normalize_embeddings=True).astype("float32")

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        texts = [self.document_prefix + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True).astype("float32")
