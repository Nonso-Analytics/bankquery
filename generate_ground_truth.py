"""
BankQuery ground truth generator.

For each FAQ record, asks an LLM to generate ~5 paraphrased,
student/customer-style questions that record would answer — using
different wording than the original question. This produces a
realistic evaluation set for retrieval (Hit Rate, MRR).

Output: data/ground_truth.csv with columns: question, document_id

Run:
    uv run python generate_ground_truth.py
"""

import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from ingest import load_faq_data

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"  # cheap, fast, good enough for paraphrasing
NUM_QUESTIONS_PER_DOC = 5
MAX_WORKERS = 6
OUTPUT_PATH = "data/ground_truth.csv"


class Questions(BaseModel):
    questions: list[str]


PROMPT_TEMPLATE = """\
You are helping build an evaluation dataset for a bank customer FAQ
search system. Below is a real FAQ from {bank} in the "{category}"
category.

Question: {question}
Answer: {answer}

Generate {n} realistic questions a bank customer might type into a
search bar or chatbot to find this exact FAQ. Requirements:
- Naturally mention "{bank}" in EVERY question (a real customer
  searching for their own bank's info would name it) — but vary how:
  sometimes at the start, sometimes at the end, sometimes woven into
  the sentence
- Use DIFFERENT wording and phrasing than the original question above
- Vary the style: some casual, some formal, some short, some longer
- Do not just reorder the same words — genuinely paraphrase
- Do not invent details not supported by the answer
- Each question should be answerable using only the answer text above
"""


def generate_questions_for_doc(doc: dict) -> list[dict]:
    """Call the LLM for one FAQ record, return list of {question, document_id} rows."""
    prompt = PROMPT_TEMPLATE.format(
        bank=doc["bank"],
        category=doc["category"],
        question=doc["question"],
        answer=doc["answer"],
        n=NUM_QUESTIONS_PER_DOC,
    )

    for attempt in range(3):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=Questions,
            )
            parsed = response.choices[0].message.parsed
            return [
                {"question": q, "document_id": doc["id"]}
                for q in parsed.questions
            ]
        except Exception as e:
            if attempt == 2:
                print(f"  FAILED doc {doc['id']} after 3 attempts: {e}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def main():
    documents = load_faq_data()
    print(f"Generating ground truth for {len(documents)} documents "
          f"({NUM_QUESTIONS_PER_DOC} questions each, ~{len(documents) * NUM_QUESTIONS_PER_DOC} total)...")

    all_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_questions_for_doc, doc): doc for doc in documents}
        for future in tqdm(as_completed(futures), total=len(futures)):
            rows = future.result()
            all_rows.extend(rows)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "document_id"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} ground truth questions -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()