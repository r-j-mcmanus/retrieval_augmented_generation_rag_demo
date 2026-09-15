from enum import StrEnum
from pydantic import BaseModel, ConfigDict, model_validator
from datetime import datetime

class DocumentScope(StrEnum):
    CLIENT = "client"
    INTERNAL = "internal"
    ALL_CLIENTS = "all_clients"


class QueryRequest(BaseModel):
    """The request json the api expects"""
    query: str
    scope: DocumentScope
    client_reference: int | None = None
    document_ids: set[int] | None = None
    created_before: datetime | None = None
    created_after: datetime | None = None

    def __str__(self) -> str:
        return self.model_dump_json()

    @model_validator(mode="after")
    def validate_scope(self):
        if self.scope == DocumentScope.CLIENT:
            if self.client_reference is None:
                raise ValueError(
                    "client_reference is required for client scope"
                )
        elif self.client_reference is not None:
            raise ValueError(
                "client_reference is only valid for client scope"
            )

        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after > self.created_before
        ):
            raise ValueError(
                "created_after must be before created_before"
            )

        return self


class DocumentFilter(BaseModel):
    """Holds the state for which documents can be searched"""
    # ensure it cannot be modified after creation
    model_config = ConfigDict(frozen=True)

    client_reference: int | None = None
    include_internal: bool = True
    # example for later dev
    client_references: frozenset[int] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
