import yaml
from pathlib import Path
import orjson

from llm_caller import LLMCallerInterface

from pydantic_dataclasses import ExtractedChunk

class KnowledgeGraph:
    def __init__(self):
        self.vertex_types: list[str] = []
        self.edge_relations: list[str]  = []
        self._load_graph_schema()

    def _load_graph_schema(self):
        relation_path = Path('knowledge_graph/valid_relation_types.yaml')
        if not relation_path.exists():
            raise Exception(f'Expected {relation_path}')
        with relation_path.open("r", encoding="utf-8") as f: 
            self.edge_relations = yaml.safe_load(f)

        vertex_path = Path('knowledge_graph/valid_vertex_types.yaml')
        if not vertex_path.exists():
            raise Exception(f'Expected {vertex_path}')
        with vertex_path.open("r", encoding="utf-8") as f: 
            self.vertex_types = yaml.safe_load(f)

    def _make_vertex_types(self):
        pass

    def _make_relationships(self):
        pass

    def add_document(self, chunks: list[ExtractedChunk], llm: LLMCallerInterface):
        prompt = self._create_prompt()
        for chunk in chunks:
            result = llm.call(prompt + '\nThe following is the context:\n' + chunk.content)
            response = result.response
            try:
                jsons = self._get_jsons(response)
            except:
                print(result)
            a=1
        a=1

    def _create_prompt(
        self
    ) -> str:
        base_string_parts: list[str] = [
            'You are a state of the art algorithm designed for extracting information in structured formats to build a knowledge graph.',
            'Your task is to identify the entities and relations requested with the user prompt from a given text.',
            'You must generate the output as a list of valid json objects.',
            'The JSON objects are of the form:',
            '{"source": "name of entity", "source_type": "entity type", "target": "name of entity", "target_type": "entity type", "relationship": "relationship type"}',

            'The "entity type" describes extracted head entity identified by the "name of entity"',
            f'The "entity type" must be from the following list: {self.vertex_types}.',

            'The "relationship type" relate to the extracted head entity identified by the "name of entity"',
            f'The "relationship type" must be from the following list: {self.edge_relations}.',
            '\n',
            'Here is an example:', # This is known as one-hot training
            '"John Doe works for Apple, and his son is Ben Doe"',
            'which returns:',
            '[{"source": "John Doe", "source_type": "Person", "target": "Apple", "target_type": "Company", "relationship": "MEMBER_OF"},',
            '{"source": "John Doe", "source_type": "Person", "target": "Ben Doe", "target_type": "Person", "relationship": "RELATED_TO"}]',
            '{"source": "Ben Doe", "source_type": "Person", "target": "John Doe", "target_type": "Person", "relationship": "RELATED_TO"}]',
            '\n',
            'Do not return the example.',
            'Be sceptical about relations and only return ones that will be of relevance to a wealth firm.',
            'Use only the listed "entity type" and "relationship type"',
            "IMPORTANT NOTES:\n- Don't add any explanation and text, just return the list of Json objects.",
        ]
        prompt = "\n".join(base_string_parts)

        return prompt

    def _get_jsons(self, text: str):
        extracted_jsons: list[str] = []
        i = 0
        while i < len(text):
            if text[i] != '{':
                i += 1
                continue

            depth = 1
            for j in range(i+1, len(text)):
                if text[j] == '{':
                    depth += 1
                if text[j] == '}':
                    depth -= 1
                if depth == 0:
                    json_string = text[i:j+1]
                    try:
                        json = orjson.loads(json_string)
                        extracted_jsons.append(json)
                    except Exception as e:
                        print(f'Cannot parse {json_string}: {e}')
                    i = j + 1
                    break

        a=1