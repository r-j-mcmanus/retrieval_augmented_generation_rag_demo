from pathlib import Path
from typing import Any

from extractors import BaseDocumentExtractor
from storage import MetadataStoreInterface, VectorStoreInterface
from embedding import EmbeddingServiceInterface
from llm_caller import LLMCallerInterface
from search_result_dataclass import SearchResult
from prompt_router import IntentRouter, IntentPath

# https://www.reddit.com/r/Rag/comments/1rf7xf6/whats_your_experience_with_hybrid_retrieval/

class RAGPipeline:
    def __init__(
        self,
        extractors: list[BaseDocumentExtractor],
        metadata_store: MetadataStoreInterface,
        vector_store: VectorStoreInterface,
        encoder: EmbeddingServiceInterface,
        llm_caller: LLMCallerInterface,
        query_router: IntentRouter
    ):
        self.extractors = {extractor.source_type.lower(): extractor for extractor in extractors}
        self.encoder = encoder
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.llm_caller = llm_caller
        self.query_router = query_router

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

    def _add_context(self, results: dict[int, SearchResult]) -> dict[int, SearchResult]:
        chunk_ids = list(results.keys())
        retrieved_chunk_data = self.metadata_store.search_by_chunk_ids(chunk_ids)

        for chunk_data in retrieved_chunk_data:
            results[chunk_data.chunk_id] = results[chunk_data.chunk_id] | chunk_data # update missing fields 

        return results

    def _combine_results(self, sparse_results: list[SearchResult], dense_results: list[SearchResult], offset: float = 60) -> dict[int, SearchResult]:
        merged_results_dict: dict[int, SearchResult] = {}

        for item in sparse_results + dense_results:
            chunk_id = int(item.chunk_id)
            if chunk_id not in merged_results_dict:
                merged_results_dict[chunk_id] = item
            else:
                existing = merged_results_dict[chunk_id]
                merged_results_dict[chunk_id] = SearchResult(
                    chunk_id = chunk_id,
                    dense_rank = item.dense_rank if item.dense_rank > -1 else existing.dense_rank,
                    sparse_rank = item.sparse_rank if item.sparse_rank > -1 else existing.sparse_rank,
                    vector_distance = item.vector_distance if item.vector_distance > -1 else existing.vector_distance,
                    key_word_score = item.key_word_score if item.key_word_score < 1 else existing.key_word_score,
                    context = item.context if item.context else existing.context
                )

        for r in merged_results_dict.values():
            if r.dense_rank >= 0:
                r.score += 1 / (offset + r.dense_rank)
            if r.sparse_rank >= 0:
                r.score += 1 / (offset + r.sparse_rank)

        return merged_results_dict

    def _build_context_parts(self, query: str, top_k: int) -> tuple[list[str], list[SearchResult]]:
        sparse_results = self.metadata_store.sparse_search(query, top_k)
        
        query_vector = self.encoder.encode_query(query)
        dense_results = self.vector_store.search(query_vector, top_k=top_k)

        combined_results = self._combine_results(sparse_results, dense_results)
        combined_results = self._add_context(combined_results)

        context_parts = []
        for r in combined_results.values():
            context_parts.append(
                f"File: {r.file_name}" \
                f" | RAG Score: {r.score}"
                # f" | Source: {hit['source_type']}" \
                # f" | Metadata: {json.dumps(hit['metadata'], default=str)}" \
                # f" | Locator: {json.dumps(hit['locator'], default=str)}" \
                f" | Content: {r.context}"
            )
        return context_parts, list(combined_results.values())

    def answer_query(self, user_query: str, top_k: int = 10) -> dict[str, Any]:
        # paths, answer = self.query_router.route(user_query, self.llm_caller)
        paths = []
        answer = ''

        context_list, matches = self._build_context_parts(user_query, top_k)
        
        # if there is no data we can retrieve relevant to the query
        if not matches:
            return {
                "query": user_query,
                "matches": [],
                "response": 'No relevant context',
                "intents": {
                    'result': [p.intent.name for p in paths],
                    'answer': answer
                }
            }

        context_answers = []
        for context in context_list: # TODO move into async loop
            context_answers.append(self._get_single_context_response(context, user_query, paths))

        match_data = [
            {
                'source': m.file_name, 
                'context': m.context, 
                'score': m.score, 
                'locator': m.locator, 
                'metadata': m.metadata, 
                'context_response': a
            } 
            for m , a in zip(matches, context_answers)
            if 'not relevant' not in a.lower()
        ]
        
        final_response = self._get_final_response(user_query, match_data)

        return {
            "query": user_query,
            "matches": match_data,
            "response": final_response,
            "intents": {
                'result': [p.intent.name for p in paths],
                'answer': answer
            }
        }

    def _get_single_context_response(self, context: str, user_query: str, paths: list[IntentPath]) -> str:
        answers = []
        for p in paths:
            answer = p.evaluate_chunk(user_query, context, self.llm_caller)
            answers.append((answer, p.intent.name))

        prompt = f"""You are analysing a contextual snippet from a business document to answer a query.
            Context Snippet:
            {context}

            User Query: {user_query}

            Intents: {[p.intent.name for p in paths]}

            Instructions:
            1. Extract ONLY facts from the snippet that answer the User Query.
            2. Do not assume or extrapolate beyond the provided text, be strict about this.
            3. Only answer using information from the context snippet.
            4. Be direct and concise, if the context is not relevant, say only that.
            
            Important note: The context may not be relevant to the query, treat it critically.
            If there is no answer relevant, respond exactly with 'not relevant'
            """
        response = self.llm_caller.call(prompt)
        return response
    
    def _get_final_response(self, user_query: str, match_data: list[dict[str, str]]) -> str:
        
        final_context = ''
        for match in match_data:
            final_context = final_context + f'\n {match['source']}: \"{match['context_response']}\".' 

        if not final_context:
            return 'No relevant sources.'

        prompt = f"""
            You are a RAG system for a private wealth management company.
            Users have provided a query and bellow is the retrieved context to augment your response.
            Synthesize a clear, direct answer to the user query using ONLY the verified source extractions provided below.

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
    