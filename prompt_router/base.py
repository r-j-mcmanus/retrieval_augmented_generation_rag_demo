from abc import ABC, abstractmethod
from .data_classes import Intent
from llm_caller import LLMCallerInterface


class IntentPath(ABC):
    """
    A query-specific strategy.

    Each path owns:
      - its intent identification
      - its prompt construction
      - its chunk evaluation
      - any query-specific post-processing
    """

    intent: Intent

    def __repr__(self) -> str:
        return f'{self.intent.name} path'
    
    @abstractmethod
    def evaluate_chunk(
        self,
        query: str,
        context: str,
        llm: LLMCallerInterface
    ) -> str:
        """Evaluate a retrieved chunk against the query."""
        raise NotImplementedError