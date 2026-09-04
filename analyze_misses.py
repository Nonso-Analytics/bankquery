"""
Look at ground-truth questions that text_search (with tuned boosts)
fails to retrieve correctly, to understand whether they're genuine
paraphrases (justifying vector search) or something else (typos,
ambiguous questions, data issues).
"""

from ingest import load_faq_data
from search import text_search, load_ground_truth

documents = load_faq_data()
doc_by_id = {d["id"]: d for d in documents}

ground_truth = load_ground_truth()

misses = []
for gt in ground_truth:
    query = gt["question"]
    expected_id = int(gt["document_id"])
    results = text_search(query, num_results=5)
    hit = any(doc["id"] == expected_id for doc in results)
    if not hit:
        misses.append((query, expected_id))

print(f"Total misses: {len(misses)} / {len(ground_truth)} ({len(misses)/len(ground_truth)*100:.1f}%)\n")

print("Sample of 15 missed questions vs. their correct answer:\n")
for query, expected_id in misses[:15]:
    correct_doc = doc_by_id[expected_id]
    print(f"Ground truth Q: {query}")
    print(f"Should match:   {correct_doc['question']}")
    print()