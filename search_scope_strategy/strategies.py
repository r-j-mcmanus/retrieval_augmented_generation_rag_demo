from .base import ScopeStrategy

from pydantic_dataclasses import QueryRequest, DocumentFilter

# these are not dependent on request so we dont need to constantly remake them
_ALL_CLIENT_DOCUMENTS_FILTER = DocumentFilter(include_internal=False)
_INTERNAL_DOCUMENTS_FILTER = DocumentFilter(include_internal=True)


class AllClientDocumentsStrategy(ScopeStrategy):
    __slots__ = ()

    def __call__(self, request: QueryRequest) -> DocumentFilter:
        return _ALL_CLIENT_DOCUMENTS_FILTER


class InternalDocumentsStrategy(ScopeStrategy):
    __slots__ = ()

    def __call__(self, request: QueryRequest) -> DocumentFilter:
        return _INTERNAL_DOCUMENTS_FILTER


class ClientDocumentsStrategy(ScopeStrategy):
    __slots__ = ()

    def __call__(self, request: QueryRequest) -> DocumentFilter:
        return DocumentFilter(
            client_reference=request.client_reference,
            include_internal=False,
        )
