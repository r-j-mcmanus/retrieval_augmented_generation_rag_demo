
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    client_reference: int | None = None
    internal: bool