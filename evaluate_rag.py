"""
BankQuery RAG answer-quality evaluation using LLM-as-a-judge.

Generates RAG answers for a sample of ground truth questions, then asks
an LLM to judge whether each generated answer is consistent with the
original FAQ answer.

Run:
    uv run python evaluate_rag.py
"""

import csv
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from ingest import load_faq_data
from rag import rag
from search import load_ground_truth

load_dotenv()
client = OpenAI()

JUDGE_MODEL = "gpt-4o-mini"
SAMPLE_SIZE = 200
MAX_WORKERS = 6
OUTPUT_PATH = "data/rag_eval_results.csv"

documents = load_faq_data()
doc_by_id = {d["id"]: d for d in documents}


class Judgment(BaseModel):
    verdict: str  # "good" or "bad"
    reasoning: str


JUDGE_PROMPT = """\
You are evaluating a RAG (Retrieval-Augmented Generation) system for a
bank FAQ assistant.

ORIGINAL QUESTION: {question}
ORIGINAL FAQ ANSWER (ground truth): {original_answer}

GENERATED ANSWER (from the RAG system): {generated_answer}

Does the generated answer correctly and faithfully convey the
information in the original FAQ answer? It doesn't need to be
word-for-word identical, but it must not contradict, omit critical
details from, or fabricate information not in the original answer.

Respond with verdict "good" or "bad", plus brief reasoning.
"""


def judge_answer(question: str, original_answer: str, generated_answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        original_answer=original_answer,
        generated_answer=generated_answer,
    )
    for attempt in range(3):
        try:
            response = client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=Judgment,
            )
            parsed = response.choices[0].message.parsed
            return {"verdict": parsed.verdict, "reasoning": parsed.reasoning}
        except Exception as e:
            if attempt == 2:
                return {"verdict": "error", "reasoning": str(e)}
            time.sleep(2 * (attempt + 1))
    return {"verdict": "error", "reasoning": "unknown"}


def evaluate_one(gt_row: dict) -> dict:
    question = gt_row["question"]
    doc_id = int(gt_row["document_id"])
    original_doc = doc_by_id[doc_id]

    rag_result = rag(question)
    generated_answer = rag_result["answer"]

    judgment = judge_answer(question, original_doc["answer"], generated_answer)

    return {
        "question": question,
        "document_id": doc_id,
        "original_answer": original_doc["answer"],
        "generated_answer": generated_answer,
        "verdict": judgment["verdict"],
        "reasoning": judgment["reasoning"],
    }


def main():
    ground_truth = load_ground_truth()
    random.seed(42)
    sample = random.sample(ground_truth, min(SAMPLE_SIZE, len(ground_truth)))

    print(f"Running RAG + judge on {len(sample)} sampled questions...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_one, row): row for row in sample}
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "document_id", "original_answer",
            "generated_answer", "verdict", "reasoning",
        ])
        writer.writeheader()
        writer.writerows(results)

    good = sum(1 for r in results if r["verdict"] == "good")
    bad = sum(1 for r in results if r["verdict"] == "bad")
    errors = sum(1 for r in results if r["verdict"] == "error")

    print(f"\nResults ({len(results)} total):")
    print(f"  Good:  {good} ({good/len(results)*100:.1f}%)")
    print(f"  Bad:   {bad} ({bad/len(results)*100:.1f}%)")
    print(f"  Errors: {errors}")
    print(f"\nSaved details -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()