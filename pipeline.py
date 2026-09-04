"""
BankQuery automated ingestion pipeline, using dlt to manage the load
step idempotently — safe to re-run without duplicating data.

Steps: load FAQ data -> embed -> load into PGVector (replace on each run).

Run:
    uv run python pipeline.py
"""

import os

import dlt
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


@dlt.resource(name="faq_documents", write_disposition="replace")
def faq_documents_resource():
    """dlt resource: yields raw FAQ documents from data/faqs.json."""
    documents = load_faq_data()
    yield from documents


def embed_and_load_to_pgvector(documents: list[dict]) -> int:
    """Embed all documents and load them into the PGVector `documents` table."""
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
    conn.close()
    return count


def ensure_conversations_table():
    """Create the conversations table (for monitoring/feedback) if it doesn't exist."""
    conn = psycopg.connect(DATABASE_URL)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            question TEXT,
            answer TEXT,
            bank_filter TEXT,
            num_sources INTEGER,
            response_time_seconds FLOAT,
            feedback INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    conn.close()
    print("conversations table ready")


def run_pipeline():
    """
    Full pipeline run:
    1. dlt tracks the raw FAQ documents as a resource (for state/lineage)
    2. Documents are embedded and loaded into PGVector (source of truth
       for the app's vector search)
    """
    documents = load_faq_data()
    print(f"Pipeline: {len(documents)} documents loaded from data/faqs.json")

    # dlt resource run — lightweight here since our "destination" for
    # vector search is PGVector directly, but this gives us dlt's
    # extract/normalize step and state tracking for the raw data layer.
    pipeline = dlt.pipeline(
        pipeline_name="bankquery_faq_ingestion",
        destination="duckdb",
        dataset_name="bankquery_raw",
    )
    load_info = pipeline.run(faq_documents_resource())
    print(f"dlt load info: {load_info}")

    count = embed_and_load_to_pgvector(documents)
    ensure_conversations_table()  
    print(f"\nPipeline complete. {count} documents embedded and loaded into PGVector.")
    return count

if __name__ == "__main__":
    run_pipeline()