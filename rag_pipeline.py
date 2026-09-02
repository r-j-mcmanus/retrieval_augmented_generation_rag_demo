from pathlib import Path
from typing import Any

from extractors import BaseDocumentExtractor
from storage import MetadataStoreInterface, VectorStoreInterface
from embedding import EmbeddingServiceInterface
from llm_caller import LLMCallerInterface


class RAGPipeline:
    def __init__(
        self,
        extractors: list[BaseDocumentExtractor],
        metadata_store: MetadataStoreInterface,
        vector_store: VectorStoreInterface,
        encoder: EmbeddingServiceInterface,
        llm_caller: LLMCallerInterface
    ):
        self.extractors = {extractor.source_type.lower(): extractor for extractor in extractors}
        self.encoder = encoder
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.llm_caller = llm_caller

    def _get_extractor_for_file(self, file_path: str | Path) -> BaseDocumentExtractor:
        suffix = Path(file_path).suffix.lower().lstrip(".")
        extractor = self.extractors.get(suffix)
        if extractor is None:
            raise ValueError(f"No extractor registered for file type: {suffix}")
        return extractor

    def index_file(self, file_path: str | Path):
        file_path = Path(file_path)
        extractor = self._get_extractor_for_file(file_path)
        chunks = extractor.extract(file_path)
        useful_metadata = extractor.get_useful_metadata(file_path)
        useful_metadata['encoder'] = self.encoder.name

        for chunk in chunks:
            chunk.metadata = useful_metadata
        
        chunk_ids = self.metadata_store.insert_document(
            file_path=file_path,
            source_type=extractor.source_type,
            metadata=useful_metadata,
            chunks=chunks,
        )

        chunk_contents = (chunk.content for chunk in chunks)
        encoded_chunk = self.encoder.encode_documents(chunk_contents)

        self.vector_store.add_vectors(chunk_ids=chunk_ids, vectors=encoded_chunk)

    def _build_context_prompt(self, query: str, top_k: int, max_distance: float) -> tuple[list[str], list[dict[str, Any]]]:
        query_vector = self.encoder.encode_query(query)
        matches = self.vector_store.search(query_vector, top_k=top_k)
        if not getattr(matches, "keys", None):
            return [], []

        chunk_ids = [int(key) for key in matches.keys]
        rows = self.metadata_store.search_by_chunk_ids(chunk_ids)
        by_chunk_id = {row["chunk_id"]: row for row in rows}

        context_parts = []
        ordered_hits = []
        for key, distance in zip(matches.keys, matches.distances):
            if distance > max_distance:
                continue
            chunk_id = int(key)
            row = by_chunk_id.get(chunk_id)
            if row is None:
                continue
            row["score"] = float(distance)
            ordered_hits.append(row)

        for hit in ordered_hits:
            context_parts.append(
                f"File: {hit['file_name']}" \
                # f" | Source: {hit['source_type']}" \
                # f" | Metadata: {json.dumps(hit['metadata'], default=str)}" \
                # f" | Locator: {json.dumps(hit['locator'], default=str)}" \
                f" | Content: {hit['content']}"
            )
        return context_parts, ordered_hits

    def answer_query(self, user_query: str, top_k: int = 4, max_distance = 0.3) -> dict[str, Any]:
        context_list, matches = self._build_context_prompt(user_query, top_k, max_distance)
        
        # if there is no data we can retrieve relevant to the query
        if not matches:
            return {
                "query": user_query,
                "matches": [],
                "response": 'No relevant context',
            }

        context_answers = []
        for context in context_list: # TODO move into async loop
            context_answers.append(self._get_single_context_response(context, user_query))

        match_data = [
            {
                'source':m.get('file_name'), 
                'context': m.get('content'), 
                'score': m.get('score'), 
                'locator': m.get('locator'), 
                'metadata': m.get('metadata'), 
                'context_response': a
            } 
            for m , a in zip(matches, context_answers)
        ]
        
        final_response = self._get_final_response(user_query, match_data)

        return {
            "query": user_query,
            "matches": match_data,
            "response": final_response,
        }

    def _get_single_context_response(self, context: str, user_query: str) -> str:
        prompt = f"""You are analysing a contextual snippet from a business document to answer a query.
            Context Snippet:
            {context}

            User Query: {user_query}

            Instructions:
            1. Extract ONLY facts from the snippet that answer the User Query.
            2. Do not assume or extrapolate beyond the provided text, be strict about this.
            3. Be direct and concise, if the context is not relevant, say only that.
            
            Important note: The context may not be relevant to the query, treat it critically.
            """
        response = self.llm_caller.call(prompt)
        return response
    
    def _get_final_response(self, user_query: str, match_data: list[dict[str, str]]) -> str:
        
        final_context = ''
        for match in match_data:
            final_context = final_context + f' \n {match['source']}: \"{match['context_response']}\".' 

        if not final_context:
            return 'No relevant sources.'

        prompt = f"""Synthesize a clear, direct answer to the user query using ONLY the verified source extractions provided below.

            Verified Source Extractions:
            {final_context}

            User Query: {user_query}

            Instructions:
            1. Combine all relevant details into a cohesive, non-redundant response.
            2. Rely solely on the extracted facts above.
            3. Maintain objective tone.
        """

        response = self.llm_caller.call(prompt)
        return response
    