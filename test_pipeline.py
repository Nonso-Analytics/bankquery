"""
Basic smoke tests for the BankQuery pipeline: confirms search functions
return well-formed results and evaluate() runs end-to-end on a small
sample. Not exhaustive unit tests — these catch pipeline breakage.

Run:
    uv run python test_pipeline.py
"""

import random

from ingest import load_faq_data
from search import text_search, evaluate, load_ground_truth
from vector_search import vector_search


def test_documents_load():
    docs = load_faq_data()
    assert len(docs) > 0, "No documents loaded"
    assert all("id" in d for d in docs), "Documents missing id field"
    assert all("bank" in d and "question" in d and "answer" in d for d in docs), \
        "Documents missing required fields"
    print(f"✓ test_documents_load: {len(docs)} documents, all well-formed")


def test_text_search_returns_results():
    results = text_search("How do I open a savings account?", num_results=5)
    assert len(results) > 0, "text_search returned no results"
    assert len(results) <= 5, "text_search returned more than requested"
    for doc in results:
        assert "question" in doc and "answer" in doc and "bank" in doc, \
            "text_search result missing required fields"
    print(f"✓ test_text_search_returns_results: {len(results)} results, well-formed")


def test_vector_search_returns_results():
    results = vector_search("How do I open a savings account?", num_results=5)
    assert len(results) > 0, "vector_search returned no results"
    assert len(results) <= 5, "vector_search returned more than requested"
    for doc in results:
        assert "question" in doc and "answer" in doc and "bank" in doc, \
            "vector_search result missing required fields"
    print(f"✓ test_vector_search_returns_results: {len(results)} results, well-formed")


def test_vector_search_bank_filter():
    results = vector_search("account opening documents", num_results=5, bank="Zenith Bank")
    assert len(results) > 0, "bank-filtered vector_search returned no results"
    assert all(doc["bank"] == "Zenith Bank" for doc in results), \
        "bank filter did not restrict results correctly"
    print(f"✓ test_vector_search_bank_filter: {len(results)} results, all Zenith Bank")


def test_evaluate_runs_on_sample():
    ground_truth = load_ground_truth()
    random.seed(0)
    sample = random.sample(ground_truth, 30)
    metrics = evaluate(vector_search, sample)
    assert "hit_rate" in metrics and "mrr" in metrics, "evaluate() missing expected keys"
    assert 0.0 <= metrics["hit_rate"] <= 1.0, "hit_rate out of range"
    assert 0.0 <= metrics["mrr"] <= 1.0, "mrr out of range"
    print(f"✓ test_evaluate_runs_on_sample: Hit Rate={metrics['hit_rate']:.3f}, MRR={metrics['mrr']:.3f}")


if __name__ == "__main__":
    print("Running BankQuery pipeline smoke tests...\n")
    test_documents_load()
    test_text_search_returns_results()
    test_vector_search_returns_results()
    test_vector_search_bank_filter()
    test_evaluate_runs_on_sample()
    print("\nAll tests passed.")