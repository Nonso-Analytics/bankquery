"""
BankQuery monitoring dashboard.

Run alongside app.py automatically (Streamlit multi-page app) —
accessible via the sidebar when running `uv run streamlit run app.py`.
"""

import os

import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="BankQuery Dashboard", page_icon="📊", layout="wide")
st.title("BankQuery Monitoring Dashboard")

DATABASE_URL = os.environ["DATABASE_URL"]


@st.cache_data(ttl=30)
def load_conversations() -> pd.DataFrame:
    conn = psycopg.connect(DATABASE_URL)
    df = pd.read_sql(
        "SELECT * FROM conversations ORDER BY created_at DESC",
        conn,
    )
    conn.close()
    return df


df = load_conversations()

if df.empty:
    st.info("No conversations logged yet. Ask a question in the main app first.")
    st.stop()

# --- Top-level metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Questions", len(df))
col2.metric("Avg Response Time", f"{df['response_time_seconds'].mean():.2f}s")

feedback_given = df[df["feedback"].notna()]
if len(feedback_given) > 0:
    positive_rate = (feedback_given["feedback"] == 1).mean() * 100
    col3.metric("Feedback Given", len(feedback_given))
    col4.metric("Positive Feedback", f"{positive_rate:.0f}%")
else:
    col3.metric("Feedback Given", 0)
    col4.metric("Positive Feedback", "—")

st.divider()

# --- Chart 1: Questions per bank filter ---
st.subheader("Questions by Bank Filter")
bank_counts = df["bank_filter"].value_counts()
st.bar_chart(bank_counts)

# --- Chart 2: Feedback ratio over time ---
st.subheader("Feedback Over Time")
if len(feedback_given) > 0:
    feedback_by_day = (
        feedback_given
        .assign(date=pd.to_datetime(feedback_given["created_at"]).dt.date)
        .groupby(["date", "feedback"])
        .size()
        .unstack(fill_value=0)
    )
    st.line_chart(feedback_by_day)
else:
    st.caption("No feedback recorded yet.")

# --- Chart 3: Response time distribution ---
st.subheader("Response Time Distribution")
st.bar_chart(df["response_time_seconds"].value_counts(bins=10).sort_index())

# --- Chart 4: Questions over time ---
st.subheader("Questions Over Time")
questions_by_day = (
    df.assign(date=pd.to_datetime(df["created_at"]).dt.date)
    .groupby("date")
    .size()
)
st.line_chart(questions_by_day)

# --- Chart 5: Number of sources returned per query ---
st.subheader("Sources Retrieved per Query")
st.bar_chart(df["num_sources"].value_counts().sort_index())

st.divider()

# --- Recent conversations table ---
st.subheader("Recent Conversations")
display_df = df[["created_at", "question", "bank_filter", "response_time_seconds", "feedback"]].head(20)
st.dataframe(display_df, use_container_width=True)