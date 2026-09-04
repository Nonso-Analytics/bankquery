"""
BankQuery: embed all FAQ documents and load them into PGVector.

Run:
    uv run python load_pgvector.py
"""

import os

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from ingest import load_faq_data

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DATABASE_URL = os.environ["DATABASE_URL"]


def vec_to_str(vector) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


def main():
    documents = load_faq_data()
    print(f"Loaded {len(documents)} documents")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [doc["question"] + " " + doc["answer"] for doc in documents]
    batch_size = 50
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        batch_vectors = model.encode(batch)
        vectors.extend(batch_vectors)

    print("Connecting to Postgres...")
    conn = psycopg.connect(DATABASE_URL)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(f"""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            bank TEXT,
            category TEXT,
            question TEXT,
            answer TEXT,
            source_url TEXT,
            embedding vector({EMBEDDING_DIM})
        )
    """)

    print("Inserting documents with embeddings...")
    for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
        conn.execute(
            """
            INSERT INTO documents (id, bank, category, question, answer, source_url, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            """,
            (
                doc["id"], doc["bank"], doc["category"],
                doc["question"], doc["answer"], doc["source_url"],
                vec_to_str(vec),
            ),
        )
    conn.commit()

    print("Building HNSW index...")
    conn.execute("""
        CREATE INDEX ON documents
        USING hnsw (embedding vector_cosine_ops)
    """)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"\nDone. {count} documents loaded into PGVector.")
    conn.close()


if __name__ == "__main__":
    main()