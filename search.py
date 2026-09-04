"""
BankQuery text search baseline using minsearch.

Wraps a minsearch keyword index over the FAQ records, plus evaluation
helpers (Hit Rate, MRR) run against the ground truth CSV. This module's
text_search() interface is the one later methods (vector, hybrid) will
match, so they're drop-in swappable.

Run:
    uv run python search.py
"""

import csv

from minsearch import Index

from ingest import load_faq_data

documents = load_faq_data()

# Build the keyword index. text_fields are searched via TF-IDF/BM25-style
# scoring; keyword_fields are exact-match filters (e.g. filtering by bank).
index = Index(
    text_fields=["question", "answer", "category"],
    keyword_fields=["bank"],
)
index.fit(documents)


BEST_BOOST = {"question": 3.0, "answer": 2.0, "category": 0.5}


def text_search(query: str, num_results: int = 5, boost: dict | None = None, bank: str | None = None) -> list[dict]:
    """Search the FAQ index with optional field boosts and bank filter."""
    filter_dict = {"bank": bank} if bank else {}
    boost_dict = boost if boost is not None else BEST_BOOST
    return index.search(
        query=query,
        filter_dict=filter_dict,
        boost_dict=boost_dict,
        num_results=num_results,
    )


def load_ground_truth(path: str = "data/ground_truth.csv") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hit_rate(relevance_list: list[list[bool]]) -> float:
    return sum(any(rel) for rel in relevance_list) / len(relevance_list)


def mrr(relevance_list: list[list[bool]]) -> float:
    total = 0.0
    for rel in relevance_list:
        for rank, is_relevant in enumerate(rel, start=1):
            if is_relevant:
                total += 1.0 / rank
                break
    return total / len(relevance_list)


def evaluate(search_function, ground_truth: list[dict]) -> dict:
    """
    Run search_function against every ground truth question, check
    whether the correct document_id appears in the results, and
    compute Hit Rate + MRR.
    """
    relevance_list = []
    for gt in ground_truth:
        query = gt["question"]
        expected_id = int(gt["document_id"])
        results = search_function(query)
        relevance = [doc["id"] == expected_id for doc in results]
        relevance_list.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_list),
        "mrr": mrr(relevance_list),
    }


if __name__ == "__main__":
    ground_truth = load_ground_truth()
    print(f"Evaluating text_search on {len(ground_truth)} ground truth questions...\n")

    baseline_metrics = evaluate(text_search, ground_truth)
    print(f"Baseline (no boosts):")
    print(f"  Hit Rate: {baseline_metrics['hit_rate']:.4f}")
    print(f"  MRR:      {baseline_metrics['mrr']:.4f}\n")

    # Grid search over field boosts
    boost_options = [0.5, 1.0, 2.0, 3.0]
    best_score = -1
    best_boost = None
    results_table = []

    for q_boost in boost_options:
        for a_boost in boost_options:
            for c_boost in boost_options:
                boost = {"question": q_boost, "answer": a_boost, "category": c_boost}

                def search_with_boost(query, b=boost):
                    return text_search(query, boost=b)

                metrics = evaluate(search_with_boost, ground_truth)
                results_table.append((boost, metrics["hit_rate"], metrics["mrr"]))

                # Use MRR as the primary metric to optimize
                if metrics["mrr"] > best_score:
                    best_score = metrics["mrr"]
                    best_boost = boost

    print("Top 5 boost combinations by MRR:")
    results_table.sort(key=lambda x: x[2], reverse=True)
    for boost, hr, mrr_score in results_table[:5]:
        print(f"  {boost} -> Hit Rate: {hr:.4f}, MRR: {mrr_score:.4f}")

    print(f"\nBest boost: {best_boost}")
    print(f"Best MRR: {best_score:.4f}")