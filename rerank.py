"""
BankQuery reranking: retrieve a wider candidate set with vector search,
then rerank with a cross-encoder for more precise top-k results.

Run:
    uv run python rerank.py
"""

from sentence_transformers import CrossEncoder

from vector_search import vector_search

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


def rerank_search(query: str, num_results: int = 5, candidate_pool: int = 10, bank: str | None = None) -> list[dict]:
    """Retrieve a wider candidate pool via vector search, then rerank with a cross-encoder."""
    candidates = vector_search(query, num_results=candidate_pool, bank=bank)
    if not candidates:
        return []

    pairs = [[query, doc["question"] + " " + doc["answer"]] for doc in candidates]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:num_results]]


if __name__ == "__main__":
    from search import load_ground_truth, evaluate
    import random

    ground_truth = load_ground_truth()
    random.seed(42)
    sample = random.sample(ground_truth, 300)

    print(f"Evaluating on {len(sample)} sampled questions...\n")

    print("Baseline (vector search only, top-5):")
    baseline_metrics = evaluate(vector_search, sample)
    print(f"  Hit Rate: {baseline_metrics['hit_rate']:.4f}")
    print(f"  MRR:      {baseline_metrics['mrr']:.4f}\n")

    print("With reranking (retrieve top-10, rerank to top-5):")
    rerank_metrics = evaluate(rerank_search, sample)
    print(f"  Hit Rate: {rerank_metrics['hit_rate']:.4f}")
    print(f"  MRR:      {rerank_metrics['mrr']:.4f}")