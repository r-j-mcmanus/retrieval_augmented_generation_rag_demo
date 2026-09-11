from .data_classes import Intent
from .base import IntentPath
from llm_caller import LLMCallerInterface
from .path_generic import GenericPath

class IntentRouter:
    """
    Selects the most appropriate IntentPath for a query.

    The router does not know how individual intents work.
    It only knows how to select among strategies.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.paths: list[IntentPath] = []
        self.generic_path = GenericPath()
        self.min_confidence = min_confidence

    def set_paths(self, paths: list[IntentPath]):
        _valid_paths = []
        for p in paths:
            if isinstance(p, GenericPath):
                print('Warning: Generic Path does not need to be passed')
            else:
                _valid_paths.append(p)
        self.paths = _valid_paths

    def _get_intent_list(self) -> list[str]:
        return [p.intent.name for p in self.paths]

    def route(self, query: str, llm: LLMCallerInterface) -> tuple[list[IntentPath], str]:
        intent_list = self._get_intent_list()
        intents = ', '.join(intent_list)

        prompt = f"""
            Query: {query}

            Intent Options: [{intents}]

            From the list of Intents Options return all that that match the query as such [...].
            If none match, return [{Intent.GENERIC.name}]
            """
        answer = llm.call(prompt)

        result = [i for i in intent_list if i.lower() in answer.lower()]
        result = result if result else [Intent.GENERIC.name]

        paths =  [p for p in self.paths if p.intent.name.lower() in answer.lower()]
        paths = paths if paths else [self.generic_path]

        return paths, answer # type: ignore
