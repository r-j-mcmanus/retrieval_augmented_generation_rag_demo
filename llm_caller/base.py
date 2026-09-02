from abc import ABC, abstractmethod


class LLMCallerInterface(ABC):
    """Interface for interacting with LLMs."""

    @abstractmethod
    def __init__(self):
        """Init method for a realised instance"""

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Hand the prompt to the LLM and return the response."""
