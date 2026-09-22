import yaml
from pathlib import Path
import orjson
from collections.abc import Iterable, Mapping
from typing import Any



from pydantic_dataclasses import ExtractedChunk, GenerateRequest

from pydantic import BaseModel, ConfigDict, ValidationError

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
    type: str
    name: str

class Edge(BaseModel):
    source: str
    target: str
    relation: str
    # confidence: float
    # document_id: int 


class KnowledgeGraph:
    def __init__(self):
        self.vertex_types: set[str] = set()
        self.edge_types: set[str]  = set()
        self.edges: dict[tuple[str, str], Edge] = dict()
        self.vertexes: dict[str, Vertex] = dict()
        self._load_graph_schema()

    def _load_graph_schema(self):
        relation_path = Path('knowledge_graph/valid_relation_types.yaml')
        if not relation_path.exists():
            raise Exception(f'Expected {relation_path}')
        with relation_path.open("r", encoding="utf-8") as f: 
            self.edge_types = set(yaml.safe_load(f))

        vertex_path = Path('knowledge_graph/valid_vertex_types.yaml')
        if not vertex_path.exists():
            raise Exception(f'Expected {vertex_path}')
        with vertex_path.open("r", encoding="utf-8") as f: 
            self.vertex_types = set(yaml.safe_load(f))

    def _make_vertex_types(self):
        pass

    def _make_relationships(self):
        pass

    def _add_to_graph(self, connections: list[_Connection]):
        for c in connections:
            if c.source not in self.vertexes:
                self.vertexes[c.source] = Vertex(
                    name = c.source,
                    type = c.source_type
                )
            if c.target not in self.vertexes:
                self.vertexes[c.target] = Vertex(
                    name = c.target,
                    type = c.target_type
                )
            if (c.source, c.target) not in self.edges:
                self.edges[(c.source, c.target)] = Edge(
                    source = c.source,
                    target = c.target,
                    relation = c.relationship
                )

    def add_document(self, chunks: list[ExtractedChunk], call):
        # TODO try and clean dirty connections 
        # TODO entity resolution 
        for chunk in chunks:
            connections = self._extract_connections_from_chunk(chunk, call)
            connections, dirty_connections = self._validate_connections(connections)
            self._add_to_graph(connections)

    def _validate_connections(
        self,
        connections: Iterable[Mapping[str, Any]],
    ) -> tuple[list[_Connection], list[_Connection]]:
        cleaned_connections: list[_Connection] = []
        dirty_connections = []
        valid_vertex_types = self.vertex_types
        valid_edge_relations = self.edge_types

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
            return self._extract_jsons_from_text(response)
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
    
    def _extract_jsons_from_text(self, text: str) -> list[dict]:
        extracted_jsons: list[dict] = []
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
                            print('Invalid json string: {json_str}')
                        i = j + 1
                        found_match = True
                        break
            
            # If no closing } then end the search
            if not found_match:
                break

        return extracted_jsons
        