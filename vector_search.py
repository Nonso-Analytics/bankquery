"""
BankQuery vector search using PGVector (Postgres + pgvector extension).

This is the production search method — vector search was selected after
evaluation showed it clearly outperforms text search and hybrid RRF on
this dataset (see README results table).

Run:
    uv run python vector_search.py
"""

import os

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from search import evaluate, load_ground_truth

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
DATABASE_URL = os.environ["DATABASE_URL"]

model = SentenceTransformer(MODEL_NAME)
conn = psycopg.connect(DATABASE_URL)


def vec_to_str(vector) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def vector_search(query: str, num_results: int = 5, bank: str | None = None) -> list[dict]:
    """Embed the query and return the top-N nearest documents from PGVector."""
    query_vector = model.encode(query)
    query_str = vec_to_str(query_vector)

    if bank:
        rows = conn.execute(
            """
            SELECT id, bank, category, question, answer, source_url
            FROM documents
            WHERE bank = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (bank, query_str, num_results),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, bank, category, question, answer, source_url
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_str, num_results),
        ).fetchall()

    return [
        {
            "id": r[0], "bank": r[1], "category": r[2],
            "question": r[3], "answer": r[4], "source_url": r[5],
        }
        for r in rows
    ]


if __name__ == "__main__":
    ground_truth = load_ground_truth()
    print(f"Evaluating PGVector vector_search on {len(ground_truth)} ground truth questions...\n")

    metrics = evaluate(vector_search, ground_truth)
    print(f"Vector search (PGVector, {MODEL_NAME}):")
    print(f"  Hit Rate: {metrics['hit_rate']:.4f}")
    print(f"  MRR:      {metrics['mrr']:.4f}")