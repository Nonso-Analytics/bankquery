"""
BankQuery RAG pipeline: search -> build prompt -> LLM answer.

Run:
    uv run python rag.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from vector_search import vector_search

load_dotenv()
client = OpenAI()

LLM_MODEL = "gpt-4o-mini"

PROMPT_TEMPLATE = """\
You are a helpful assistant answering customer questions about Nigerian
bank products and services, using ONLY the FAQ context provided below.

RULES:
- Answer using the information in the context. If different banks have
  different answers, clearly attribute each fact to its bank rather
  than stating it as universally true.
- Do not add requirements, steps, fees, or details not explicitly
  stated in the context, even if they seem reasonable.
- If the context doesn't contain enough information, say so honestly.

CONTEXT:
{context}

QUESTION: {question}

Give a direct, well-attributed answer.
"""


def build_context(search_results: list[dict]) -> str:
    parts = []
    for doc in search_results:
        parts.append(
            f"[{doc['bank']} — {doc['category']}]\n"
            f"Q: {doc['question']}\n"
            f"A: {doc['answer']}"
        )
    return "\n\n".join(parts)


def build_prompt(question: str, search_results: list[dict]) -> str:
    context = build_context(search_results)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def rag(question: str, num_results: int = 5, bank: str | None = None) -> dict:
    """Run the full RAG pipeline, returning the answer plus metadata."""
    search_results = vector_search(question, num_results=num_results, bank=bank)
    prompt = build_prompt(question, search_results)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    return {
        "question": question,
        "answer": answer,
        "search_results": search_results,
        "usage": response.usage,
    }


if __name__ == "__main__":
    test_question = "What documents do I need to open a savings account?"
    result = rag(test_question)
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print(f"Sources used:")
    for doc in result["search_results"]:
        print(f"  - [{doc['bank']}] {doc['question']}")