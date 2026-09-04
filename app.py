"""
BankQuery Streamlit app.

Run:
    uv run streamlit run app.py
"""

import os
import time

import psycopg
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from vector_search import vector_search

load_dotenv()

st.set_page_config(page_title="BankQuery", page_icon="🏦", layout="centered")

LLM_MODEL = "gpt-4o-mini"
DATABASE_URL = os.environ["DATABASE_URL"]

BANKS = ["Fidelity Bank", "Zenith Bank", "GTBank", "First Bank", "UBA"]

client = OpenAI()


@st.cache_resource
def get_db_connection():
    return psycopg.connect(DATABASE_URL, autocommit=True)


def build_context(search_results: list[dict]) -> str:
    parts = []
    for doc in search_results:
        parts.append(
            f"[{doc['bank']} — {doc['category']}]\n"
            f"Q: {doc['question']}\n"
            f"A: {doc['answer']}"
        )
    return "\n\n".join(parts)


NORMAL_PROMPT = """\
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

COMPARISON_PROMPT = """\
You are a helpful assistant comparing Nigerian bank products and
services, using ONLY the FAQ context provided below, which contains
entries from multiple banks.

Structure your answer as a clear comparison, one section or bullet per
bank, using ONLY information explicitly present in the context. If a
bank has no relevant information in the context, say so rather than
guessing or leaving it out silently.

CONTEXT:
{context}

QUESTION: {question}

Give a structured, bank-by-bank comparison.
"""


def answer_question(question: str, bank_filter: str | None, comparison_mode: bool) -> dict:
    start = time.time()

    if comparison_mode:
        # Retrieve per-bank so every bank gets a fair shot at appearing,
        # rather than one bank dominating the top-k globally.
        all_results = []
        for bank in BANKS:
            all_results.extend(vector_search(question, num_results=2, bank=bank))
        search_results = all_results
        prompt_template = COMPARISON_PROMPT
    else:
        search_results = vector_search(question, num_results=5, bank=bank_filter)
        prompt_template = NORMAL_PROMPT

    context = build_context(search_results)
    prompt = prompt_template.format(context=context, question=question)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    elapsed = time.time() - start

    return {
        "answer": answer,
        "search_results": search_results,
        "response_time": elapsed,
    }


def log_conversation(question: str, answer: str, bank_filter: str | None,
                      num_sources: int, response_time: float) -> int:
    conn = get_db_connection()
    row = conn.execute(
        """
        INSERT INTO conversations (question, answer, bank_filter, num_sources, response_time_seconds)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (question, answer, bank_filter, num_sources, response_time),
    ).fetchone()
    return row[0]


def log_feedback(conversation_id: int, feedback: int):
    conn = get_db_connection()
    conn.execute(
        "UPDATE conversations SET feedback = %s WHERE id = %s",
        (feedback, conversation_id),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("BankQuery")
st.caption("Ask about account opening, fees, documents, and digital banking across 5 Nigerian banks.")

with st.sidebar:
    st.header("Options")
    comparison_mode = st.toggle("Compare across all banks", value=False)
    bank_filter = None
    if not comparison_mode:
        bank_choice = st.selectbox("Filter to one bank", ["All banks"] + BANKS)
        bank_filter = None if bank_choice == "All banks" else bank_choice

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

question = st.text_input("Your question", placeholder="What documents do I need to open a savings account?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Searching and generating answer..."):
        result = answer_question(question, bank_filter, comparison_mode)

    st.markdown("### Answer")
    st.write(result["answer"])

    conversation_id = log_conversation(
        question=question,
        answer=result["answer"],
        bank_filter="ALL (comparison)" if comparison_mode else (bank_filter or "All banks"),
        num_sources=len(result["search_results"]),
        response_time=result["response_time"],
    )
    st.session_state.conversation_id = conversation_id

    with st.expander(f"Sources ({len(result['search_results'])})"):
        for doc in result["search_results"]:
            st.markdown(f"**[{doc['bank']}]** {doc['question']}")
            st.caption(doc["answer"])

    st.caption(f"Answered in {result['response_time']:.1f}s")

if st.session_state.conversation_id:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Helpful"):
            log_feedback(st.session_state.conversation_id, 1)
            st.success("Thanks for the feedback!")
    with col2:
        if st.button("👎 Not helpful"):
            log_feedback(st.session_state.conversation_id, -1)
            st.info("Thanks — noted.")