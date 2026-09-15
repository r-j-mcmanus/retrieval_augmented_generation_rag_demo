from abc import ABC, abstractmethod

from pydantic_dataclasses import QueryRequest, DocumentFilter

class ScopeStrategy(ABC):
    __slots__ = ()

    @abstractmethod
    def __call__(self, request: QueryRequest) -> DocumentFilter:
        ...
        