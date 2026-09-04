"""
Evaluate whether query rewriting improves vector search Hit Rate/MRR,
compared to searching with the raw query.

Run:
    uv run python evaluate_query_rewrite.py
"""

import random

from query_rewrite import rewrite_query
from search import load_ground_truth, evaluate
from vector_search import vector_search

SAMPLE_SIZE = 300


def vector_search_with_rewrite(query: str, num_results: int = 5) -> list[dict]:
    rewritten = rewrite_query(query)
    return vector_search(rewritten, num_results=num_results)


if __name__ == "__main__":
    ground_truth = load_ground_truth()
    random.seed(42)
    sample = random.sample(ground_truth, min(SAMPLE_SIZE, len(ground_truth)))

    print(f"Evaluating on {len(sample)} sampled questions...\n")

    print("Baseline (raw query, vector search):")
    baseline_metrics = evaluate(vector_search, sample)
    print(f"  Hit Rate: {baseline_metrics['hit_rate']:.4f}")
    print(f"  MRR:      {baseline_metrics['mrr']:.4f}\n")

    print("With query rewriting:")
    rewrite_metrics = evaluate(vector_search_with_rewrite, sample)
    print(f"  Hit Rate: {rewrite_metrics['hit_rate']:.4f}")
    print(f"  MRR:      {rewrite_metrics['mrr']:.4f}")