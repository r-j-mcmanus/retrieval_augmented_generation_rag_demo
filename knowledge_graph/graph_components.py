
from pydantic import BaseModel


class Vertex(BaseModel):
    id: str
    type: str
    name: str

class Edge(BaseModel):
    source: str
    target: str
    relation: str
    confidence: float
    relational_chunk_id: int
    document: str
    document_id: int 
