from .base import LLMCallerInterface


class RemoteLLMCaller(LLMCallerInterface):
    """Makes an external api call."""

    def __init__(self, *args, **kwags):
        """Init method for a realised instance"""
        super().__init__(*args, **kwags)

    def call(self, prompt: str) -> str:
        """Hand the prompt to the LLM and return the response."""
        return 'Not Implemented'
