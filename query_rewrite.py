"""
BankQuery query rewriting: expands vague/short user queries into more
retrieval-friendly phrasings before search, using a cheap LLM call.

Run:
    uv run python query_rewrite.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

REWRITE_MODEL = "gpt-4o-mini"

REWRITE_PROMPT = """\
You are helping a search engine understand vague banking questions.
Rewrite the user's question to be clearer and more specific for
retrieval purposes, without changing its meaning or adding facts.

- If the question is already clear and specific, return it unchanged.
- Expand vague phrasing into more explicit terms (e.g. "what do I need
  to open an account" -> "what documents and requirements are needed to
  open a bank account").
- Keep it short — one sentence.
- Do not answer the question, only rewrite it.

Original question: {question}

Rewritten question:"""


def rewrite_query(question: str) -> str:
    response = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    test_questions = [
        "what do I need to open an account",
        "fees?",
        "how much can I withdraw at Zenith Bank ATM",
    ]
    for q in test_questions:
        rewritten = rewrite_query(q)
        print(f"Original:  {q}")
        print(f"Rewritten: {rewritten}\n")