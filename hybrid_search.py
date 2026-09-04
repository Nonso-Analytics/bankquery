"""
BankQuery hybrid search using Reciprocal Rank Fusion (RRF) over
text_search and vector_search results.

Run:
    uv run python hybrid_search.py
"""

from search import text_search, evaluate, load_ground_truth
from vector_search import vector_search


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Combine multiple ranked result lists into one, using RRF:
    score(doc) = sum over lists of 1 / (k + rank_in_that_list)

    Docs are identified by their 'id' field.
    """
    scores: dict[int, float] = {}
    doc_by_id: dict[int, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]
            doc_by_id[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return [doc_by_id[doc_id] for doc_id in ranked_ids]


def hybrid_search(query: str, num_results: int = 5, bank: str | None = None) -> list[dict]:
    text_results = text_search(query, num_results=10, bank=bank)
    vector_results = vector_search(query, num_results=10, bank=bank)
    fused = reciprocal_rank_fusion([text_results, vector_results])
    return fused[:num_results]


if __name__ == "__main__":
    ground_truth = load_ground_truth()
    print(f"Evaluating hybrid_search on {len(ground_truth)} ground truth questions...\n")

    metrics = evaluate(hybrid_search, ground_truth)
    print(f"Hybrid search (RRF: text + vector):")
    print(f"  Hit Rate: {metrics['hit_rate']:.4f}")
    print(f"  MRR:      {metrics['mrr']:.4f}")