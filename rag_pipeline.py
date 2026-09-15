from pathlib import Path
from typing import Any
from dataclasses import dataclass

from extractors import BaseDocumentExtractor
from storage import MetadataStoreInterface, VectorStoreInterface
from embedding import EmbeddingServiceInterface
from llm_caller import LLMCallerInterface
from prompt_router import IntentRouter, IntentPath
from reranking import ReRanker
from text_preprocessing import TextPreprocessor
from knowledge_graph import KnowledgeGraph

from pydantic_dataclasses import QueryRequest, SearchResult, TokenUsage, LLMResponse, ExtractedChunk
from search_scope_strategy import strategy_selector

# https://www.reddit.com/r/Rag/comments/1rf7xf6/whats_your_experience_with_hybrid_retrieval/

# TODO would be good to fine tune a model for this!

class RAGPipeline:
    def __init__(
        self,
        extractors: list[BaseDocumentExtractor],
        metadata_store: MetadataStoreInterface,
        vector_store: VectorStoreInterface,
        encoder: EmbeddingServiceInterface,
        llm_caller: LLMCallerInterface,
        query_router: IntentRouter,
        re_ranker: ReRanker,
        preprocessor: TextPreprocessor,
        knowledge_graph: KnowledgeGraph
    ):
        self.extractors = {extractor.source_type.lower(): extractor for extractor in extractors}
        self.encoder = encoder
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.llm_caller = llm_caller
        self.query_router = query_router
        self.re_ranker = re_ranker
        self.preprocessor = preprocessor
        self.knowledge_graph = knowledge_graph

    def _get_extractor_for_file(self, file_path: str | Path) -> BaseDocumentExtractor:
        suffix = Path(file_path).suffix.lower().lstrip(".")
        extractor = self.extractors.get(suffix)
        if extractor is None:
            raise ValueError(f"No extractor registered for file type: {suffix}")
        return extractor

    def index_file(self, file_path: str | Path, client_reference: int | None):
        file_path = Path(file_path)
        extractor = self._get_extractor_for_file(file_path)
        chunks, useful_metadata = extractor.extract_document(
            file_path,
            additional_metadata={"encoder": self.encoder.name},
        )

        self.knowledge_graph.add_document(chunks, self.llm_caller)

        return

        chunk_ids = self.metadata_store.insert_document(
            file_path=file_path,
            source_type=extractor.source_type,
            client_reference=client_reference,
            metadata=useful_metadata,
            chunks=chunks,
        )

        chunk_contents = (self.preprocessor(chunk.content) for chunk in chunks)
        encoded_chunk = self.encoder.encode_documents(chunk_contents)

        self.vector_store.add_vectors(chunk_ids=chunk_ids, vectors=encoded_chunk)

    def _add_context(
        self,
        results: dict[int, SearchResult],
        client_reference: int | None = None,
    ) -> dict[int, SearchResult]:
        chunk_ids = list(results.keys())
        retrieved_chunk_data = self.metadata_store.search_by_chunk_ids(
            chunk_ids,
            client_reference=client_reference,
        )

        valid_chunk_ids = {chunk_data.chunk_id for chunk_data in retrieved_chunk_data}
        results = {
            chunk_id: result
            for chunk_id, result in results.items()
            if chunk_id in valid_chunk_ids
        }

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

    def _build_context_parts(
        self,
        query_request: QueryRequest,
        top_k: int,
        retrieval_number: int = 20,
    ) -> tuple[list[str], list[SearchResult]]:
        # as not all calls can access all files
        document_filter = strategy_selector(query_request)

        # todo append client ref to vector table
        allowed_chunk_ids = self.metadata_store.get_chunk_ids(document_filter)

        if allowed_chunk_ids == set():
            return [], []

        sparse_results = self.metadata_store.sparse_search(query_request, allowed_chunk_ids, retrieval_number)

        processed_query = self.preprocessor(query_request.query)
        query_vector = self.encoder.encode_query(processed_query)
        dense_results = self.vector_store.search(
            query_vector,
            top_k=retrieval_number,
            allowed_chunk_ids=allowed_chunk_ids,
        )

        combined_results = self._combine_results(sparse_results, dense_results)
        combined_results = self._add_context(combined_results, query_request.client_reference)

        top_combined_results = self.re_ranker.condense(query_request.query, list(combined_results.values()), top_k)

        context_parts = []
        for r in top_combined_results:
            context_parts.append(
                f"RAG Score: {r.score} | Content: {r.context}"
            )
        return context_parts, top_combined_results

    def answer_query(
        self,
        query_request: QueryRequest,
        top_k: int = 10,
    ) -> dict[str, Any]:
        # TODO real validation, probably before pipeline
        assert isinstance(query_request, QueryRequest)

        # paths, answer = self.query_router.route(user_query, self.llm_caller)
        paths = []
        answer = ''

        user_query = query_request.query
        query_request.query = self.preprocessor(query_request.query)

        context_list, matches = self._build_context_parts(query_request, top_k)
        
        # if there is no data we can retrieve relevant to the query
        if not matches:
            return {
                "query": user_query,
                "cleaned_query": query_request.query,
                "matches": [],
                "response": 'No relevant context',
                "intents": {
                    'result': [p.intent.name for p in paths],
                    'answer': answer
                }
            }

        context_answers: list[LLMResponse] = []
        for context in context_list: # TODO move into async loop
            llm_response = self._get_single_context_response(context, user_query, paths)
            context_answers.append(llm_response)

        match_data = [
            {
                'match': match,
                'context_response': llm_response
            } 
            for match , llm_response in zip(matches, context_answers)
            if 'not relevant' not in llm_response.response.lower()
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

    def _get_single_context_response(self, context: str, user_query: str, paths: list[IntentPath]) -> LLMResponse:
        answers = []
        for p in paths:
            answer = p.evaluate_chunk(user_query, context, self.llm_caller)
            answers.append((answer, p.intent.name))
            # Intents: {[p.intent.name for p in paths]}

        prompt = f"""You are analysing a document snippet to answer a query for a private wealth management firm.
            Context Snippet:
            `{context}`

            User Query: `{user_query}`

            Instructions:
            1. Extract ONLY facts from the snippet relevant to the User Query.
            
            If there is no relevant information, respond with 'not relevant'
            """
        response = self.llm_caller.call(prompt)
        return response
    
    def _get_final_response(self, user_query: str, match_data: list[dict[str, Any]]) -> LLMResponse:
        
        final_context = ''
        for m in match_data:
            final_context = final_context + f'\n {m['match'].file_name}: \"{m['context_response'].response}\".' 

        if not final_context:
            return LLMResponse(response='No relevant sources.', token_usage=TokenUsage(prompt_tokens=0,completion_tokens=0))

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
    