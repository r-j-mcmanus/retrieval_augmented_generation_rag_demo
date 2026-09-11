from __future__ import annotations

from .data_classes import Intent
from .base import IntentPath
from llm_caller import LLMCallerInterface


class NumericalPath(IntentPath):
    intent = Intent.NUMERICAL

    def __init__(self):
        pass

    def evaluate_chunk(
        self,
        query: str,
        context: str,
        llm: LLMCallerInterface
    ) -> str:

        result = llm.call(
            f"""
            You are extracting numerical evidence from a business document.

            Query:
            {query}

            Context:
            {context}

            Determine whether the context contains evidence that directly
            answers the query.

            Pay particular attention to:
            - metric
            - numerical value
            - unit
            - reporting period
            - entity
            - qualifiers

            Do not infer missing information.
            """
        )

        return result
