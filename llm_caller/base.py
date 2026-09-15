from abc import ABC, abstractmethod

from pydantic_dataclasses import LLMResponse


class LLMCallerInterface(ABC):
    """Interface for interacting with LLMs."""

    @abstractmethod
    def __init__(self):
        """Init method for a realised instance"""

    @abstractmethod
    def call(self, prompt: str) -> LLMResponse:
        """Hand the prompt to the LLM and return the response."""
