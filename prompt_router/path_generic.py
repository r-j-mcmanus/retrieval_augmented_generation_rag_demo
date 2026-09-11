from __future__ import annotations

from .data_classes import Intent
from .base import IntentPath
from llm_caller import LLMCallerInterface


class GenericPath(IntentPath):
    intent = Intent.GENERIC

    def __init__(self):
        pass

    def evaluate_chunk(
        self,
        query: str,
        context: str,
        llm: LLMCallerInterface
    ) -> str:

        result = llm.call(
            f"""You are analysing a contextual snippet from a business document to answer a query.
            Context Snippet:
            {context}

            User Query: {query}

            Instructions:
            1. Extract ONLY facts from the snippet that answer the User Query.
            2. Do not assume or extrapolate beyond the provided text, be strict about this.
            3. Only answer using information from the context snippet.
            4. Be direct and concise, if the context is not relevant, say only that.
            
            Important note: The context may not be relevant to the query, treat it critically.

            """
        )

        return result



