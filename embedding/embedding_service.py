from typing import Iterable

from sentence_transformers import SentenceTransformer
import numpy as np

from .base import EmbeddingServiceInterface


class _MODELS:
    bge_base_en_v1_5 = "BAAI/bge-base-en-v1.5"
    bge_small_en_v1_5 = "BAAI/bge-small-en-v1.5"


class BGEEmbeddingService(EmbeddingServiceInterface):
    """Uses a BGE model and sentence transformer for the encoding."""

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5"):
        super().__init__(model, 768)
        self.model = SentenceTransformer(model, cache_folder="./_local_models") # if we wanted it local
        self.query_prefix = "Represent this sentence for searching relevant passages: "
        self.document_prefix = "Represent this document for retrieval: "

    def encode_query(self, query: str) -> np.ndarray:
        query = self.query_prefix + query
        return self.model.encode(query, normalize_embeddings=True).astype("float32")

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        texts = [self.document_prefix + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True).astype("float32")
