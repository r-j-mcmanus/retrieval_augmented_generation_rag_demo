from __future__ import annotations

from .data_classes import Intent
from .base import IntentPath
from llm_caller import LLMCallerInterface


class ComparisonPath(IntentPath):
    intent = Intent.COMPARISON

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
            Evaluate this context as evidence for a comparison query.

            Query:
            {query}

            Context:
            {context}

            The context may contain evidence about only one side of the
            comparison. Preserve partial evidence rather than rejecting it.

            Do not infer information not present in the context.
            """
        )

        return result
