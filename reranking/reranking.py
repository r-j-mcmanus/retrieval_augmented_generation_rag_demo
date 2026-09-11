from sentence_transformers import CrossEncoder

from search_result_dataclass import SearchResult

class ReRanker:
    def __init__(
        self,
        model_name: str = 'BAAI/bge-reranker-base',
        cache_folder: str = './_local_models',
        device: str | None = None,
    ) -> None:
        self.model = CrossEncoder(
            model_name,
            cache_folder=cache_folder,
            device=device,
        )

    def condense(self, query: str, search_results: list[SearchResult], top_k: int, min_score: float = 0.001) -> list[SearchResult]:
        if top_k < 0:
            raise ValueError('top_k must be non-negative')
        if top_k == 0 or not search_results:
            return []

        pairs = [(query, result.context) for result in search_results]
        scores = self.model.predict(pairs)

        ranked_results = sorted(
            zip(search_results, scores),
            key=lambda result_and_score: result_and_score[1],
            reverse=True,
        )

        condensed_results = []
        for result, score in ranked_results:
            if score < min_score:
                break
            result.score = float(score)
            condensed_results.append(result)
            if len(condensed_results) == top_k:
                break

        return condensed_results
    