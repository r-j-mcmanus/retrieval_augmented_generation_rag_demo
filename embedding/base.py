from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class EmbeddingServiceInterface(ABC):
    """Shared encoder used by all extractors once chunks are built."""

    def __init__(self, name: str, embedding_dim: int):
        """Responsible for encoding text into a latent vector allowing for retrieval"""
        self.name = name
        self.embedding_dim = embedding_dim
    
    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        """Encodes the past iterable into latent vectors"""

    @abstractmethod
    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Encodes the past iterable into latent vectors"""
