import yaml
import json
from uuid import UUID, uuid4
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, ValidationError, Field

from pydantic_dataclasses import ExtractedChunk, GenerateRequest


class _Connection(BaseModel): 
    """Raw data extracted from document"""
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source: str
    source_type: str
    target: str
    target_type: str
    relationship: str

class Vertex(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    type: str
    name: str
    client_reference: str | None = None

class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    relation: str
    client_reference: str | None = None
    # confidence: float
    # document_id: int 


class KnowledgeGraph:
    _database_path = Path("./database")
    _edges_path = _database_path / "edges.json"
    _vertexes_path = _database_path / "vertexes.json"
    _vertex_ids_by_client_reference_path = _database_path / "vertex_ids_by_client_reference.json"

    def __init__(self):
        self.vertex_types: list[dict[str, str]] = []
        self.edge_types: list[dict[str, str]] = []
        self.edges: dict[UUID, Edge] = dict()
        self.vertexes: dict[UUID, Vertex] = dict()
        self.vertex_ids_by_client_reference: dict[str, set[UUID]] = dict()
        self._load_graph_schema()
        self._load_graph()

    def _load_graph(self):
        if self._edges_path.exists():
            edge_data = json.loads(self._edges_path.read_text(encoding="utf-8"))
            self.edges = {
                UUID(edge_id): Edge.model_validate(edge)
                for edge_id, edge in edge_data.items()
            }

        if self._vertexes_path.exists():
            vertex_data = json.loads(self._vertexes_path.read_text(encoding="utf-8"))
            self.vertexes = {
                UUID(vertex_id): Vertex.model_validate(vertex)
                for vertex_id, vertex in vertex_data.items()
            }

        if self._vertex_ids_by_client_reference_path.exists():
            client_vertex_data = json.loads(
                self._vertex_ids_by_client_reference_path.read_text(encoding="utf-8")
            )
            self.vertex_ids_by_client_reference = {
                client_reference: {UUID(vertex_id) for vertex_id in vertex_ids}
                for client_reference, vertex_ids in client_vertex_data.items()
            }

    def _save_graph(self):
        self._database_path.mkdir(parents=True, exist_ok=True)
        self._edges_path.write_text(
            json.dumps(
                {str(edge_id): edge.model_dump(mode="json") for edge_id, edge in self.edges.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        self._vertexes_path.write_text(
            json.dumps(
                {str(vertex_id): vertex.model_dump(mode="json") for vertex_id, vertex in self.vertexes.items()},
                indent=2,
            ),
            encoding="utf-8",
        )
        self._vertex_ids_by_client_reference_path.write_text(
            json.dumps(
                {
                    client_reference: sorted(str(vertex_id) for vertex_id in vertex_ids)
                    for client_reference, vertex_ids in self.vertex_ids_by_client_reference.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_graph_schema(self):
        relation_path = Path('knowledge_graph/valid_relation_types.yaml')
        if not relation_path.exists():
            raise Exception(f'Expected {relation_path}')
        with relation_path.open("r", encoding="utf-8") as f: 
            self.edge_types = yaml.safe_load(f)

        vertex_path = Path('knowledge_graph/valid_vertex_types.yaml')
        if not vertex_path.exists():
            raise Exception(f'Expected {vertex_path}')
        with vertex_path.open("r", encoding="utf-8") as f: 
            self.vertex_types = yaml.safe_load(f)

    def _make_vertex_types(self):
        pass

    def _make_relationships(self):
        pass

    def _add_to_graph(
        self,
        connections: list[_Connection],
        client_reference: str | None = None,
    ):
        for c in connections:
            v_s =  Vertex(
                name = c.source,
                type = c.source_type,
                client_reference = client_reference,
            )
            self.vertexes[v_s.id] = v_s

            v_t =  Vertex(
                name = c.target,
                type = c.target_type,
                client_reference = client_reference,
            )
            self.vertexes[v_t.id] = v_t

            if client_reference is not None:
                client_vertices = self.vertex_ids_by_client_reference.setdefault(
                    client_reference, set()
                )
                client_vertices.update((v_s.id, v_t.id))

            e = Edge(
                    source_id = v_s.id,
                    target_id = v_t.id,
                    relation = c.relationship,
                    client_reference = client_reference,
            )
            self.edges[e.id] = e

    def add_document(
        self,
        chunks: list[ExtractedChunk],
        call,
        client_reference: str | None = None,
    ):
        # TODO try and clean dirty connections 
        # TODO entity resolution 
        for chunk in chunks:
            connections = self._extract_connections_from_chunk(chunk, call)
            connections, dirty_connections = self._validate_connections(connections)
            self._add_to_graph(connections, client_reference)
        self._save_graph()

    def _validate_connections(
        self,
        connections: Iterable[Mapping[str, Any]],
    ) -> tuple[list[_Connection], list[_Connection]]:
        cleaned_connections: list[_Connection] = []
        dirty_connections = []
        valid_vertex_types = [i['entity'] for i in self.vertex_types]
        valid_edge_relations = [i['relation'] for i in self.edge_types]

        for candidate in connections:
            try:
                valid_connection = _Connection.model_validate(candidate)
            except ValidationError:
                dirty_connections.append(candidate)
                continue

            if (valid_connection.source_type not in valid_vertex_types):
                dirty_connections.append(candidate)
                continue
            if (valid_connection.target_type not in valid_vertex_types):
                dirty_connections.append(candidate)
                continue
            if (valid_connection.relationship not in valid_edge_relations):
                dirty_connections.append(candidate)
                continue

            cleaned_connections.append(valid_connection)
                
        return cleaned_connections, dirty_connections

    def _extract_connections_from_chunk(self, chunk: ExtractedChunk, call) -> list[dict]:
        request = self._create_prompt(chunk.content)
        result = call(request.query, request.system_prompt)
        response = result.response
        try:
            extracted_jsons, invalid_strs = self._extract_jsons_from_text(response)
            print(f'invalid json strings: {invalid_strs}')
            return extracted_jsons
        except Exception:
            print(result)
            return []

    def _create_prompt(self, content: str) -> GenerateRequest:
        return GenerateRequest(system_prompt="""
You extract entities and relations from text to build a knowledge graph.

Return ONLY a JSON array of objects. Each object must have exactly these fields:

{
"source": "name of entity",
"source_type": "entity_type",
"target": "name of entity",
"target_type": "entity_type",
"relationship": "relationship_type"
}

for example:
[{
    "source": "John Doe",
    "source_type": "PERSON",
    "target": "£1 million",
    "target_type": "MORTGAGE",
    "relationship": "HAS_MORTGAGE"
},{
    "source": "Alice Peters",
    "source_type": "PERSON",
    "target": "child's mortgage",
    "target_type": "MORTGAGE",
    "relationship": "HAS_GOAL"
}
]

Use ONLY the allowed entity types and relationship types provided by the user.
Do not invent entity types or relationship types.
Do not include explanations, markdown, or additional text.
            """,
            query=f"""
Extract the entities and relations from the following context.

<context>
{content}
</context>

Return only the JSON array.
The "entity type" must ONLY be from the following list: <entity_type>{list(self.vertex_types)}</entity_type>.
The "relationship type" must ONLY be from the following list: <relationship_type>{list(self.edge_types)}</relationship_type>.
""")
    
    def _extract_jsons_from_text(self, text: str) -> tuple[list[dict], list[str]]:
        extracted_jsons: list[dict] = []
        invalid_strs: list[str] = []
        i = 0
        n = len(text)
        
        while i < n:
            i = text.find('{', i)# Jump directly to the next '{'
            if i == -1:
                break

            depth = 1
            found_match = False
            
            for j in range(i + 1, n):
                char = text[j]
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = text[i:j + 1]
                        try:
                            json_dict = orjson.loads(json_str)
                            extracted_jsons.append(json_dict)
                        except Exception as e:
                            invalid_strs.append(json_str)
                        i = j + 1
                        found_match = True
                        break
            
            # If no closing } then end the search
            if not found_match:
                break

        return extracted_jsons, invalid_strs
        