"""
BankQuery ingestion module.

Loads the consolidated FAQ dataset and prepares it for indexing.
Mirrors the LLM Zoomcamp's load_faq_data() pattern.
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "faqs.json"


def load_faq_data() -> list[dict]:
    """
    Load data/faqs.json and return a list of document dicts, each with
    a stable integer id, ready for indexing.

    Each document has:
      id, bank, category, question, answer, source_url, scraped_at
    """
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    documents = []
    for i, item in enumerate(raw):
        doc = dict(item)  # copy
        doc["id"] = i
        documents.append(doc)

    return documents


if __name__ == "__main__":
    docs = load_faq_data()
    print(f"Loaded {len(docs)} documents")
    print("Sample:", docs[0])