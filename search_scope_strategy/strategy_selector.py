from pydantic_dataclasses import QueryRequest
from .base import ScopeStrategy
from .strategies import AllClientDocumentsStrategy
from .strategies import InternalDocumentsStrategy
from .strategies import ClientDocumentsStrategy

from pydantic_dataclasses import DocumentScope, DocumentFilter


_STRATEGIES : dict[DocumentScope, ScopeStrategy]= {
    DocumentScope.ALL_CLIENTS: AllClientDocumentsStrategy(),
    DocumentScope.INTERNAL: InternalDocumentsStrategy(),
    DocumentScope.CLIENT: ClientDocumentsStrategy(),
}


def strategy_selector(request: QueryRequest) -> DocumentFilter:
    return _STRATEGIES[request.scope](request)
