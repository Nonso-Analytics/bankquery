"""
Evaluates the minsearch text-search baseline against the ground truth
using Hit Rate and MRR.

Also runs a small grid search over field boosts — don't assume `question`
should be boosted highest, let the numbers decide.

Run:
    uv run python evaluate_search.py
"""

import pandas as pd
from tqdm.auto import tqdm

from ingest import load_faq_data, build_index


def hit_rate(relevance: list[list[int]]) -> float:
    cnt = 0
    for line in relevance:
        if 1 in line:
            cnt += 1
    return cnt / len(relevance)


def mrr(relevance: list[list[int]]) -> float:
    total_score = 0.0
    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance)


def compute_relevance(q: dict, search_function) -> list[int]:
    doc_id = q["document"]
    results = search_function(query=q["question"])
    return [int(d["id"] == doc_id) for d in results]


def compute_relevance_total(ground_truth: list[dict], search_function) -> list[list[int]]:
    relevance_total = []
    for q in tqdm(ground_truth):
        relevance_total.append(compute_relevance(q, search_function))
    return relevance_total


def evaluate(ground_truth: list[dict], search_function) -> dict:
    relevance_total = compute_relevance_total(ground_truth, search_function)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


def main():
    documents = load_faq_data()
    index = build_index(documents)

    df_ground_truth = pd.read_csv("data/ground_truth.csv")
    ground_truth = df_ground_truth.to_dict(orient="records")
    print(f"Loaded {len(ground_truth)} ground-truth questions")

    # --- Baseline: no boosts ---
    def text_search_default(query):
        return index.search(query, num_results=5)

    print("\nEvaluating baseline (no boosts)...")
    baseline = evaluate(ground_truth, text_search_default)
    print(f"Baseline: {baseline}")

    # --- Grid search over boosts ---
    print("\nGrid-searching field boosts...")

    def search_with_boosts(query, question_boost, answer_boost, category_boost):
        boost_dict = {
            "question": question_boost,
            "answer": answer_boost,
            "category": category_boost,
        }
        return index.search(query, num_results=5, boost_dict=boost_dict)

    grid_results = []
    for question_boost in [0.5, 1.0, 2.0, 3.0]:
        for answer_boost in [1.0, 2.0, 4.0]:
            for category_boost in [0.1, 0.5, 1.0]:
                result = evaluate(
                    ground_truth,
                    lambda query, qb=question_boost, ab=answer_boost, cb=category_boost:
                        search_with_boosts(query, qb, ab, cb)
                )
                grid_results.append({
                    "question_boost": question_boost,
                    "answer_boost": answer_boost,
                    "category_boost": category_boost,
                    "hit_rate": result["hit_rate"],
                    "mrr": result["mrr"],
                })

    df_grid = pd.DataFrame(grid_results)
    df_grid = df_grid.sort_values("mrr", ascending=False)
    df_grid.to_csv("data/boost_grid_search.csv", index=False)

    print("\nTop 10 boost combinations by MRR:")
    print(df_grid.head(10).to_string(index=False))

    best = df_grid.iloc[0]
    print(f"\nBest: question={best['question_boost']}, answer={best['answer_boost']}, "
          f"category={best['category_boost']} -> hit_rate={best['hit_rate']:.3f}, mrr={best['mrr']:.3f}")

    print("\nSaved full grid to data/boost_grid_search.csv")
    print("\nNext: hardcode the winning boosts into a text_search() function in ingest.py or a new search.py,")
    print("then look at which ground-truth questions still fail — that's what tells you whether vector search is worth adding (see TASKS.md 2.4).")


if __name__ == "__main__":
    main()