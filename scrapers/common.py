"""
Shared schema + helpers for all BankQuery scrapers.

Every scraper (fidelity.py, zenith.py, gtbank.py, firstbank.py, uba.py)
should produce a list of dicts matching the common schema and save it
with save_faqs(). This keeps every bank's output consistent for ingestion.
"""

import json
import re
import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FAQ_SCHEMA_FIELDS = ["bank", "category", "question", "answer", "source_url", "scraped_at"]


def make_faq(bank: str, category: str, question: str, answer: str, source_url: str) -> dict:
    """Build a single FAQ record in the common schema."""
    return {
        "bank": bank.strip(),
        "category": clean_text(category),
        "question": clean_text(question),
        "answer": clean_text(answer),
        "source_url": source_url.strip(),
        "scraped_at": datetime.date.today().isoformat(),
    }


def clean_text(text: str) -> str:
    """Normalize whitespace and strip junk characters commonly picked up from HTML."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dedupe(faqs: list[dict]) -> list[dict]:
    """Remove exact duplicate (bank, question) pairs, keeping the first occurrence."""
    seen = set()
    out = []
    for f in faqs:
        key = (f["bank"], f["question"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def save_faqs(faqs: list[dict], bank_slug: str) -> Path:
    """Save one bank's FAQs to data/<bank_slug>.json"""
    faqs = dedupe(faqs)
    out_path = DATA_DIR / f"{bank_slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(faqs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(faqs)} FAQs for {bank_slug} -> {out_path}")
    return out_path


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

def merge_all(bank_slugs: list[str]) -> Path:
    """Combine all per-bank JSON files into a single data/faqs.json"""
    all_faqs = []
    for slug in bank_slugs:
        path = DATA_DIR / f"{slug}.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            all_faqs.extend(json.load(f))

    out_path = DATA_DIR / "faqs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_faqs, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(all_faqs)} total FAQs -> {out_path}")
    return out_path